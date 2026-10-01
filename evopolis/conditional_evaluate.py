"""Task 05 matched conditional-response outcomes with paired human-group inference.

Individual prequential scores and freely generated endpoint scores remain
separate prediction targets. The declared joint claim is their intersection.
"""
import argparse
from collections import defaultdict
import csv
import datetime
import fcntl
import json
import math
import resource
import tempfile
import time

import numpy as np

from .behavior_data import ROOT, content_hash, load, write_json
from .behavior_evaluate import write_csv
from .forecast_evaluate import (FORECAST_METRICS, MECHANISMS, energy_score, crps,
                                 interval_coverage, renewal_score, score_cell)
from .sources import sha256

RESULTS = ROOT / 'results/task05'
SEEDS = (17, 29, 43)
FAMILIES = ('P0', 'P1', 'H0', 'H1')
BOOTSTRAP_SEED = 20261022
BOOTSTRAP_REPLICATES = 2000
CONTRASTS = (('H1', 'H0'), ('P1', 'P0'), ('H1', 'P1'), ('H0', 'P0'))


def procedure(row):
    return row['family'] if row['family'] in FAMILIES else f"{row['family']}_{row['budget']}"


def paired_group_summary(rows, metrics, *, mechanisms=MECHANISMS,
                         replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED):
    """Seed means inside complete keys; equal mechanism means and paired draws."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[(procedure(row), row['mechanism'], int(row['group_index']))].append(row)
    procedures = sorted({procedure(row) for row in rows})
    group_ids = {m: sorted({int(r['group_index']) for r in rows if r['mechanism'] == m}) for m in mechanisms}
    keys = {int(r['group_index']): r.get('group_key', str(int(r['group_index']))) for r in rows}
    for row in rows:
        if row.get('group_key', str(int(row['group_index']))) != keys[int(row['group_index'])]:
            raise ValueError('One group index maps to differing full human keys')
    available = [m for m in mechanisms if group_ids[m]]
    if not procedures:
        return {'summary': [], 'comparisons': [], 'group_counts': {m: 0 for m in mechanisms},
                'unavailable_mechanisms': list(mechanisms), 'all_declared_mechanisms_available': False}
    values = {}
    for mechanism in available:
        values[mechanism] = np.empty((len(procedures), len(group_ids[mechanism]), len(metrics)))
        for p, name in enumerate(procedures):
            for g, group in enumerate(group_ids[mechanism]):
                selected = grouped[(name, mechanism, group)]
                if len(selected) != 3 or {int(r['seed']) for r in selected} != set(SEEDS):
                    raise ValueError(f'Missing/duplicate fitted seeds: {name}/{mechanism}/{group}')
                values[mechanism][p, g] = [math.fsum(float(r[metric]) for r in selected)/3 for metric in metrics]
    rng = np.random.Generator(np.random.PCG64(seed))
    bootstrap = np.zeros((replicates, len(procedures), len(metrics)))
    indices = {}
    for mechanism in available:
        indices[mechanism] = rng.integers(len(group_ids[mechanism]), size=(replicates, len(group_ids[mechanism])))
        bootstrap += values[mechanism][:, indices[mechanism], :].mean(axis=2).transpose(1, 0, 2)/len(available)
    overall = np.mean([values[m].mean(axis=1) for m in available], axis=0)
    summary = []
    for p, name in enumerate(procedures):
        for mechanism in mechanisms:
            means = values[mechanism][p].mean(axis=0) if mechanism in available else None
            summary.append({'procedure': name, 'mechanism': mechanism, 'groups': len(group_ids[mechanism]),
                            **{metric: float(means[i]) if means is not None else None for i, metric in enumerate(metrics)}})
        entry = {'procedure': name, 'mechanism': 'all' if len(available) == len(mechanisms) else 'available_mechanisms_only',
                 'groups': sum(map(len, group_ids.values())), **{metric: float(overall[p, i]) for i, metric in enumerate(metrics)},
                 'ci95': {metric: np.quantile(bootstrap[:, p, i], (.025, .975)).tolist() for i, metric in enumerate(metrics)}}
        per_seed = []
        for fitted_seed in SEEDS:
            selected = [r for r in rows if procedure(r) == name and int(r['seed']) == fitted_seed]
            per_seed.append({'seed': fitted_seed, **{metric: float(np.mean([np.mean([float(r[metric]) for r in selected if r['mechanism'] == m]) for m in available])) for metric in metrics}})
        entry['per_seed'] = per_seed
        summary.append(entry)
    comparisons = []
    for new_name, old_name in (*CONTRASTS, ('recurrent_continued', 'recurrent_original'), ('feedforward_continued', 'feedforward_original')):
        if new_name not in procedures or old_name not in procedures:
            continue
        new, old = procedures.index(new_name), procedures.index(old_name)
        comparisons.append({'comparison': f'{new_name} minus {old_name}', 'new': new_name, 'reference': old_name,
                            'primary_contrast': (new_name, old_name) == ('H1', 'H0'), 'groups': sum(map(len, group_ids.values())),
                            'estimand': 'equal mechanisms' if len(available) == len(mechanisms) else 'available mechanisms only',
                            'differences': {metric: float(overall[new, i]-overall[old, i]) for i, metric in enumerate(metrics)},
                            'ci95': {metric: np.quantile(bootstrap[:, new, i]-bootstrap[:, old, i], (.025, .975)).tolist() for i, metric in enumerate(metrics)},
                            'by_mechanism': {m: {metric: float((values[m][new, :, i]-values[m][old, :, i]).mean()) for i, metric in enumerate(metrics)} if m in available else None for m in mechanisms},
                            'per_seed': [{'seed': fitted_seed, **{metric: next(r for r in summary if r['procedure'] == new_name and 'per_seed' in r)['per_seed'][j][metric]-next(r for r in summary if r['procedure'] == old_name and 'per_seed' in r)['per_seed'][j][metric] for metric in metrics}} for j, fitted_seed in enumerate(SEEDS)]})
    plan = {'group_ids': group_ids, 'group_keys': {m: [keys[i] for i in group_ids[m]] for m in mechanisms},
            'resampled_indices': {m: value.tolist() for m, value in indices.items()}}
    return {'aggregation': 'Scores per seed, mean seeds within complete human group, mean groups within mechanism, mechanisms equally; never pool predictive draws across fits',
            'summary': summary, 'comparisons': comparisons, 'group_counts': {m: len(group_ids[m]) for m in mechanisms},
            'unavailable_mechanisms': [m for m in mechanisms if m not in available], 'all_declared_mechanisms_available': len(available) == len(mechanisms),
            'bootstrap': {'replicates': replicates, 'seed': seed, 'generator': 'PCG64', 'unit': 'human interacting group', 'paired': True,
                          'group_ids': group_ids, 'group_keys': plan['group_keys'], 'plan_sha256': content_hash(plan),
                          'scope': 'Human sample uncertainty conditional on these selected fits and finite forecast banks; no claim to all parameter uncertainty'}}


def freeze_evaluation():
    from .conditional_generate import checkpoint_registry, validate_integration
    registry = checkpoint_registry()
    validate_integration(registry)
    contract = {'config_sha256': sha256(ROOT / 'configs/task05.json'), 'scoring_code_sha256': sha256(ROOT / 'evopolis/conditional_evaluate.py'),
                'model_code_sha256': sha256(ROOT / 'evopolis/conditional_models.py'),
                'split_sha256': sha256(ROOT / 'results/task03/split.json'), 'checkpoints': registry,
                'integration_sha256': sha256(RESULTS / 'integration_checks.json'),
                'bootstrap_seed': BOOTSTRAP_SEED, 'bootstrap_replicates': BOOTSTRAP_REPLICATES}
    path = RESULTS / 'evaluation_opening.json'
    if path.exists() and json.loads(path.read_text())['contract'] != contract:
        raise ValueError('Frozen evaluation contract changed; document a correctness repair before rescoring')
    if not path.exists():
        write_json(path, {'opened_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                          'status': 'Diagnostic use of already opened Experiment 1; not an independently confirmatory test', 'contract': contract})
    return contract


def _base(spec, group):
    return {'family': spec['family'], 'budget': spec['budget'], 'seed': spec['seed'],
            'selected_epoch': spec['selected_epoch'], 'checkpoint_sha256': spec['sha256'],
            'group_index': group['index'], 'group_key': json.dumps(group['key'], separators=(',', ':')), 'mechanism': group['mechanism']}


def _score_observation_group(spec, group, arrays, probabilities, log_prob, individual, window, collective, rounds, zero_windows):
    index = group['index']
    base = _base(spec, group)
    offers = arrays['offers'][index].T
    mask = offers >= 1
    legal = np.floor(offers).astype(np.int64)
    targets = arrays['y'][index].T
    count = int(mask.sum())
    if count:
        individual.append({**base, 'nonforced_choices': count, 'nll': float(-log_prob[mask].mean())})
    else:
        zero_windows.append({**base, 'window': 'full game', 'nonforced_choices': 0})
    primary = group['mechanism'] in MECHANISMS and mask[5].any()
    if primary:
        window_mask = mask[5:15]
        if window_mask.any():
            window.append({**base, 'source_round_start': 5, 'source_round_stop_exclusive': 15,
                           'nonforced_choices': int(window_mask.sum()), 'nll': float(-log_prob[5:15][window_mask].mean())})
        else:
            zero_windows.append({**base, 'window': '5:15', 'nonforced_choices': 0})
    if group['mechanism'] not in MECHANISMS:
        return
    live_rounds = np.flatnonzero(mask.any(axis=1))
    scores = []
    for t in live_rounds:
        score = renewal_score(probabilities[t], legal[t], offers[t], targets[t])
        scores.append(score)
        rounds.append({**base, 'round_id': int(t), 'round_weight': 1/(3*8*len(live_rounds)), **score})
    if scores:
        collective.append({**base, 'nonforced_rounds': len(live_rounds),
                           **{name: float(np.mean([r[name] for r in scores])) for name in ('aggregate_nll', 'renewal_brier', 'renewal_probability', 'renewal_observed')}})


def _calibration_bins(rounds):
    bins = []
    for name in sorted({procedure(r) for r in rounds}):
        for mechanism in (*MECHANISMS, 'all'):
            selected = [r for r in rounds if procedure(r) == name and (mechanism == 'all' or r['mechanism'] == mechanism)]
            for b in range(10):
                low, high = b/10, (b+1)/10
                records = [r for r in selected if low <= r['renewal_probability'] and (r['renewal_probability'] <= high if b == 9 else r['renewal_probability'] < high)]
                weight = math.fsum(r['round_weight'] for r in records)
                bins.append({'procedure': name, 'mechanism': mechanism, 'bin': b, 'lower': low, 'upper': high,
                             'weighted_mass': weight if mechanism != 'all' else weight/3,
                             'predicted': math.fsum(r['renewal_probability']*r['round_weight'] for r in records)/weight if weight else None,
                             'observed': math.fsum(r['renewal_observed']*r['round_weight'] for r in records)/weight if weight else None,
                             'group_count': len({r['group_index'] for r in records}), 'round_seed_count': len(records)})
    for name in sorted({procedure(r) for r in rounds}):
        for mechanism in (*MECHANISMS, 'all'):
            if not math.isclose(math.fsum(r['weighted_mass'] for r in bins if r['procedure'] == name and r['mechanism'] == mechanism), 1., abs_tol=1e-12):
                raise ValueError('Calibration bin weights do not sum to one')
    return bins


def evaluate_observations():
    import torch
    from .behavior_train import runtime_setup
    from .conditional_generate import checkpoint_registry, load_checkpoint
    from .conditional_models import prequential
    from .behavior_models import emission_log_prob, emission_probs
    from .forecast_generate import checkpoint_registry as neural_registry, load_checkpoint as load_neural
    runtime_setup()
    freeze_evaluation()
    started, cpu = time.monotonic(), time.process_time()
    arrays, manifest = load()
    groups = [g for g in manifest['groups'] if g['split'] == 'test']
    individual, window, collective, rounds, zero_windows = [], [], [], [], []
    log_records, identities = [], []
    for spec in checkpoint_registry():
        model, _ = load_checkpoint(spec)
        fit_log_probs = []
        for group in groups:
            with torch.no_grad():
                prediction = prequential(model, arrays, group['index'], nodes=spec['integration_nodes'])
            _score_observation_group(spec, group, arrays, prediction['pmfs'], prediction['log_prob'], individual, window, collective, rounds, zero_windows)
            fit_log_probs.append(prediction['log_prob'])
        log_records.append(fit_log_probs)
        identities.append(spec)
        print(f"Conditional observations: {spec['family']}/{spec['seed']}", flush=True)
    # Frozen old procedures retain their original nine features and histories.
    for spec in neural_registry():
        if spec['family'] not in ('feedforward', 'recurrent'):
            continue
        model, _ = load_neural(spec)
        fit_log_probs = []
        for group in groups:
            index = group['index']
            with torch.no_grad():
                raw, _ = model(torch.tensor(arrays['x'][index], dtype=torch.float32))
                n = torch.tensor(arrays['n'][index])
                y = torch.tensor(arrays['y'][index])
                log_prob = emission_log_prob(raw, n, y).numpy().T
                probabilities = emission_probs(raw, n).numpy().transpose(1, 0, 2)
            _score_observation_group(spec, group, arrays, probabilities, log_prob, individual, window, collective, rounds, zero_windows)
            fit_log_probs.append(log_prob)
        log_records.append(fit_log_probs)
        identities.append(spec)
        print(f"Frozen neural observations: {spec['family']}/{spec['seed']}/{spec['budget']}", flush=True)
    old_lookup = {(r['family'], r['budget'], int(r['seed']), int(r['group_index'])): float(r['nll']) for r in csv.DictReader((ROOT / 'results/task04/individual_groups.csv').open())}
    disagreements = [abs(r['nll']-old_lookup[(r['family'], r['budget'], r['seed'], r['group_index'])]) for r in individual if r['family'] not in FAMILIES]
    if max(disagreements) > 1e-6:
        raise ValueError('Frozen reference NLL differs from its Task 04 value')
    primary = paired_group_summary(window, ('nll',))
    if primary['group_counts'] != {'Equal': 5, 'Mixed': 8, 'Proportional': 8} or zero_windows:
        raise ValueError('Primary individual cohort or nonforced windows differ; document before claiming identical cohort')
    summary = {'status': 'Already opened Experiment 1; no reserved transfer outcomes accessed',
               'individual_32': paired_group_summary(individual, ('nll',), mechanisms=(*MECHANISMS, 'M1')),
               'individual_primary_window': primary,
               'collective_one_step': paired_group_summary(collective, ('aggregate_nll', 'renewal_brier', 'renewal_probability', 'renewal_observed')),
               'zero_choice_groups_or_windows': zero_windows, 'frozen_reference_max_nll_disagreement': max(disagreements),
               'primary_window': 'Warm on rounds 0..4, then predict before updating each source round 5..14',
               'neural_comparison_limit': 'Old neural predictors use original nine inputs; not information matched to new conditional families',
               'renewal_event': '1.4 * total returns >= sum(recorded offers)',
               'resident_independence': 'Convolution assumes conditional independent resident choices; does not empirically establish independence',
               'resource': {'wall_seconds': time.monotonic()-started, 'cpu_seconds': time.process_time()-cpu,
                            'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}}
    for name, rows in (('individual_groups.csv', individual), ('individual_primary_window.csv', window),
                       ('collective_one_step_groups.csv', collective), ('collective_calibration_rounds.csv', rounds),
                       ('collective_calibration_bins.csv', _calibration_bins(rounds))):
        write_csv(RESULTS / name, rows)
    with tempfile.NamedTemporaryFile(dir=RESULTS, suffix='.npz', delete=False) as sink:
        temporary = sink.name
        np.savez_compressed(sink, log_prob=np.asarray(log_records))
    from pathlib import Path
    Path(temporary).replace(RESULTS / 'test_prequential_log_probs.npz')
    write_json(RESULTS / 'test_prequential_identity.json', {
        'axes': 'record, group, source round, resident', 'record_order': identities, 'groups': groups,
        'source_sha256': manifest['source_sha256'], 'split_sha256': sha256(ROOT / 'results/task03/split.json'),
        'array_sha256': sha256(RESULTS / 'test_prequential_log_probs.npz')})
    write_json(RESULTS / 'one_step_summary.json', summary)
    return summary


def _reference_forecasts(manifest):
    group_lookup = {g['index']: g for g in manifest['groups']}
    references = []
    text_fields = {'cell_id', 'family', 'budget', 'checkpoint_sha256', 'mechanism', 'bank', 'convention'}
    int_fields = {'seed', 'group_index', 'origin', 'horizon', 'eligible_at_origin', 'branches'}
    for row in csv.DictReader((ROOT / 'results/task04/forecast_scores.csv').open()):
        if row['family'] not in ('feedforward', 'recurrent') or row['convention'] != 'recorded':
            continue
        typed = {name: value if name in text_fields else int(value) if name in int_fields else float(value) for name, value in row.items()}
        typed['group_key'] = json.dumps(group_lookup[typed['group_index']]['key'], separators=(',', ':'))
        references.append(typed)
    return references


def evaluate_forecasts():
    from .conditional_generate import iter_cells, ARCHIVE
    freeze_evaluation()
    started, cpu = time.monotonic(), time.process_time()
    arrays, manifest = load()
    rows = []
    count = 0
    for meta, body, paths in iter_cells():
        scored = score_cell(meta, body, paths, arrays)
        for row in scored:
            row['group_key'] = json.dumps(meta['key'], separators=(',', ':'))
        rows.extend(scored)
        count += 1
        if count % 200 == 0:
            print(f'Conditional forecast scores: {count}/1404 cells', flush=True)
    if count != 1404:
        raise ValueError('Incomplete forecast archive')
    reference_rows = _reference_forecasts(manifest)
    all_rows = rows+reference_rows
    primary_window = list(csv.DictReader((RESULTS / 'individual_primary_window.csv').open()))
    lookup = {(r['family'], r['budget'], int(r['seed']), int(r['group_index'])): r for r in primary_window}
    def primary_rows(bank):
        chosen = [r for r in all_rows if r['origin'] == 5 and r['horizon'] == 10 and r['eligible_at_origin'] and r['bank'] == bank]
        combined = []
        for row in chosen:
            individual = lookup[(row['family'], row['budget'], row['seed'], row['group_index'])]
            if row['group_key'] != individual['group_key']:
                raise ValueError('Individual and collective full group keys differ')
            combined.append({**row, 'nll': float(individual['nll']), 'nonforced_choices': int(individual['nonforced_choices'])})
        return combined
    metrics = ('nll', *FORECAST_METRICS)
    primary = paired_group_summary(primary_rows('main'), metrics)
    second = paired_group_summary(primary_rows('second'), metrics)
    if primary['group_counts'] != {'Equal': 5, 'Mixed': 8, 'Proportional': 8}:
        raise ValueError('Primary collective cohort differs')
    if primary['bootstrap']['plan_sha256'] != second['bootstrap']['plan_sha256']:
        raise ValueError('Primary and stability bank used different human resampling')
    saved_individual = json.loads((RESULTS / 'one_step_summary.json').read_text())['individual_primary_window']
    if primary['bootstrap']['plan_sha256'] != saved_individual['bootstrap']['plan_sha256']:
        raise ValueError('Individual and collective endpoints used different human resampling')
    secondary = []
    for origin in (0, 5, 10, 20):
        for horizon in (1, 5, 10, 20):
            selected = [r for r in all_rows if r['origin'] == origin and r['horizon'] == horizon and r['bank'] == 'main']
            for population in ('all_groups', 'origin_eligible'):
                subset = selected if population == 'all_groups' else [r for r in selected if r['eligible_at_origin']]
                secondary.append({'origin': origin, 'horizon': horizon, 'population': population,
                                  **paired_group_summary(subset, FORECAST_METRICS)})
    effect = next(r for r in primary['comparisons'] if r['comparison'] == 'H1 minus H0')
    joint = all(effect['ci95'][metric][1] < 0 for metric in ('nll', 'energy'))
    summary = {'primary': {'origin': 5, 'horizon': 10, 'population': 'origin eligible', 'bank': 'principal 64 draws', **primary},
               'second_bank': {'origin': 5, 'horizon': 10, 'bank': 'independent second 64 draws; not pooled', **second},
               'secondary': secondary, 'joint_predictive_improvement': joint,
               'decision_rule': 'Joint predictive improvement requires both H1-minus-H0 primary 95% intervals below zero; positive peer response is an additional interpretation condition',
               'cell_count': count, 'branch_count': count*64, 'score_rows': len(rows),
               'energy_estimator': 'Off-diagonal Euclidean energy score; no clipping; endpoint pool/200 and total window retained resources/(200*h)',
               'path_reporting': 'Temporally ordered pool and surplus ensemble trajectories saved; trajectory_pool_crps is a mean marginal score, not a joint path score',
               'coverage': 'Inclusive observed endpoint coverage in empirical quantiles [0.1,0.9], NumPy linear interpolation',
               'frozen_neural_references': {'source': 'results/task04/forecast_scores.csv', 'sha256': sha256(ROOT / 'results/task04/forecast_scores.csv'),
                                            'limit': 'Different original nine-input information set; new four-family comparisons are matched'},
               'archive_sha256': sha256(ARCHIVE),
               'resource': {'wall_seconds': time.monotonic()-started, 'cpu_seconds': time.process_time()-cpu,
                            'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}}
    write_csv(RESULTS / 'forecast_scores.csv', rows)
    write_csv(RESULTS / 'primary_paired_groups.csv', primary_rows('main'))
    write_csv(RESULTS / 'second_bank_paired_groups.csv', primary_rows('second'))
    write_json(RESULTS / 'forecast_summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('observations', 'forecasts', 'all'))
    args = parser.parse_args()
    with (ROOT / 'data/cache/task03/training.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.action in ('observations', 'all'):
            evaluate_observations()
        if args.action in ('forecasts', 'all'):
            evaluate_forecasts()


if __name__ == '__main__':
    main()
