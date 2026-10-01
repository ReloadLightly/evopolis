"""Task 06 closed-loop training-group calibration and fixed tilt sensitivity.

Every grid point uses the same indexed random numbers within checkpoint/rule.
Atomic point receipts make the full fixed grid resumable without changing it.
"""
import argparse
import itertools
import json
import resource
import time

import numpy as np

from .behavior_data import ROOT, content_hash, write_json
from .institutional_metrics import energy_distance, outcome_vectors, summarize
from .institutional_rollout import RESULTS, registry, load_checkpoint, seeds_for, simulate
from .sources import sha256

RULES = ('Equal', 'Mixed', 'Proportional')


def calibration():
    config = json.loads((ROOT/'configs/task06.json').read_text())
    spec = config['calibration']
    human = json.loads((RESULTS/'recorded_games.json').read_text())['games']
    target = {r: outcome_vectors([g for g in human if g['cohort'] == 'Exp1' and
               g['split'] == 'train' and g['mechanism'] == r]) for r in RULES}
    records = [r for r in registry() if r['family'] in config['calibration_families']]
    coarse = list(itertools.product(spec['tau0_grid'], spec['tau1_grid']))
    folder = RESULTS/'calibration_points'
    folder.mkdir(exist_ok=True)
    start = time.monotonic()
    selected = {}
    timings = []
    unique_point_durations = {}
    for record in records:
        key = f"{record['family']}_{record['seed']}"
        model = load_checkpoint(record)
        seeds = {r: seeds_for(record['family'], record['seed'], r, spec['games_per_rule'],
                             config['calibration_namespace']) for r in RULES}
        contract = {'checkpoint': record, 'config_sha256': sha256(ROOT/'configs/task06.json'),
                    'code_sha256': {n: sha256(ROOT/n) for n in
                       ('evopolis/institutional_rollout.py', 'evopolis/institutional_models.py',
                        'evopolis/institutional_calibrate.py', 'evopolis/institutional_metrics.py')},
                    'human_training_targets': content_hash({r: v.tolist() for r,v in target.items()}),
                    'seeds': {r: [str(s) for s in values] for r, values in seeds.items()}}

        def point(pair):
            pair = tuple(round(x, 10) for x in pair)
            identifier = content_hash({'key': key, 'tilt': pair})[:24]
            path = folder/f'{key}_{identifier}.json'
            if path.exists():
                result = json.loads(path.read_text())
                if result['contract'] != contract or result['tilt'] != list(pair):
                    raise ValueError('Changed calibration point contract')
            else:
                tick = time.monotonic()
                cells = {}
                for rule in RULES:
                    rows, _, audit = simulate(model, rule, seeds[rule], tilt=pair, retain=0)
                    cells[rule] = {'energy': energy_distance(outcome_vectors(rows), target[rule]),
                                   'summary': summarize(rows), 'audit': audit}
                result = {'contract': contract, 'tilt': list(pair), 'cells': cells,
                          'objective': sum(v['energy'] for v in cells.values()),
                          'seconds': time.monotonic()-tick,
                          'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
                write_json(path, result)
            timings.append(result['seconds'])
            unique_point_durations[str(path)] = result['seconds']
            # There are 97 scheduled evaluations per fit. Repeated center/grid
            # coordinates reuse exact point receipts; this projection is conservative.
            projection = float(np.mean(timings))*len(records)*(len(coarse)+25)
            progress = {'scheduled_points_done': len(timings), 'scheduled_points_total': len(records)*97,
                        'projected_seconds': projection, 'seconds_this_invocation': time.monotonic()-start,
                        'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                        'current_checkpoint': key, 'last_tilt': list(pair)}
            write_json(RESULTS/'calibration_progress.json', progress)
            if len(timings) % 8 == 0 or len(timings) == 1:
                print(f"Calibration {key} point {len(timings)}/{len(records)*97}: {result['objective']:.5f}; projected {projection/3600:.2f}h", flush=True)
            if projection > config['phase_limit_seconds']:
                write_json(RESULTS/'calibration_blocker.json', dict(progress,
                    blocker='Measured calibration projection exceeds the declared eight-hour limit'))
                raise RuntimeError('Calibration projected beyond eight hours; exact grid point receipts saved')
            return result

        candidates = [point(pair) for pair in coarse]
        best = min(candidates, key=lambda v: v['objective'])
        center = best['tilt']
        refined = [point((center[0]+a, center[1]+b)) for a,b in itertools.product(
                      spec['refinement_tau0_offsets'], spec['refinement_tau1_offsets'])]
        best = min(refined, key=lambda v: v['objective'])
        selected[key] = {'tilt': best['tilt'], 'objective': best['objective'],
                         'coarse_best': center, 'checkpoint_sha256': record['sha256']}
        write_json(RESULTS/'calibration.json', {'selected': selected, 'complete': len(selected) == 18,
                    'selection_uses': 'Exp1 training only, three rules, energy-distance fidelity',
                    'seconds_this_invocation': time.monotonic()-start,
                    'unique_point_compute_seconds': sum(unique_point_durations.values()),
                    'unique_points': len(unique_point_durations),
                    'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})


def sensitivity():
    from .institutional_predict import predict_group
    from .behavior_data import load
    config = json.loads((ROOT/'configs/task06.json').read_text())
    arrays, manifest = load()
    groups = [g for g in manifest['groups'] if g['split'] == 'validation' and g['mechanism'] in RULES]
    records = registry(True, True)
    taus = config['sensitivity']['taus']
    folder = RESULTS/'sensitivity'
    folder.mkdir(exist_ok=True)
    timings = []
    for record in records:
        model = load_checkpoint(record)
        for tau in taus:
            tick = time.monotonic()
            key = f"{record['family']}_{record['seed']}_{tau:+.1f}"
            path = folder/f'{key}.json'
            tilt = [record['tilt'][0]+tau, record['tilt'][1]]
            contract = {'record': record, 'tau': tau, 'tilt': tilt,
                        'config_sha256': sha256(ROOT/'configs/task06.json'),
                        'code_sha256': {n: sha256(ROOT/n) for n in
                           ('evopolis/institutional_rollout.py', 'evopolis/institutional_predict.py',
                            'evopolis/institutional_models.py', 'evopolis/institutional_calibrate.py')}}
            if path.exists():
                saved = json.loads(path.read_text())
                if saved['contract'] != contract:
                    raise ValueError('Sensitivity contract changed')
            else:
                cells = {}
                for rule in RULES:
                    seeds = seeds_for(record['family'], record['seed'], rule, 256, config['rollout_namespace'])
                    rows, _, _ = simulate(model, rule, seeds, tilt=tilt, retain=0)
                    predictions = [predict_group(model, arrays, g['index'], tilt=tilt) for g in groups if g['mechanism'] == rule]
                    cells[rule] = {'simulation': summarize(rows),
                                   'validation': {name: float(np.mean([v[name] for v in predictions])) for name in
                                                  ('nll', 'predicted_fraction', 'observed_fraction')}}
                saved = {'contract': contract, 'cells': cells, 'seconds': time.monotonic()-tick}
                write_json(path, saved)
            timings.append(saved['seconds'])
            projection = float(np.mean(timings))*len(records)*len(taus)
            write_json(RESULTS/'sensitivity_progress.json', {'completed': len(timings),
                       'total': len(records)*len(taus), 'projected_seconds': projection,
                       'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
            print(f'Sensitivity {key}; projected {projection/3600:.2f}h', flush=True)
            if projection > config['phase_limit_seconds']:
                raise RuntimeError('Sensitivity projected beyond eight hours; cells saved')


if __name__ == '__main__':
    from .behavior_train import runtime_setup
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['calibrate', 'sensitivity'])
    args = parser.parse_args()
    runtime_setup()
    (calibration if args.command == 'calibrate' else sensitivity)()
