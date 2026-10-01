"""Independent saved-evidence audit; never imports EvoPolis scoring helpers.

SciPy's condensed pair distances and a sorted scalar-CRPS identity check the
primary endpoint scores. A separate group-difference bootstrap checks intervals.
"""
from collections import defaultdict
import csv
import datetime
import fcntl
import hashlib
import io
import json
import math
from pathlib import Path
import sqlite3
import time
import resource
import zlib

import numpy as np
from scipy.spatial.distance import pdist

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results/task04'
MECHANISMS = ('Equal', 'Mixed', 'Proportional')


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b''):
            value.update(chunk)
    return value.hexdigest()


def verify_preservation_and_selection():
    preserved = json.loads((RESULTS / 'preservation.json').read_text())
    for filename, expected in preserved['files'].items():
        if digest(ROOT / filename) != expected:
            raise AssertionError(f'Previously published artifact changed: {filename}')
    selections = []
    for family in ('feedforward', 'recurrent'):
        for seed in (17, 29, 43):
            parent = ROOT / 'results/task03/weights' / f'{family}_{seed}'
            folder = RESULTS / 'weights' / f'{family}_{seed}'
            original = json.loads((parent / 'training.json').read_text())
            logs = json.loads((folder / 'training.json').read_text())
            metadata = json.loads((folder / 'metadata.json').read_text())
            if len(original) != 120 or len(logs) != 480 or logs[:120] != original:
                raise AssertionError('Continuation lost original complete learning curve')
            if [row['epoch'] for row in logs] != list(range(1,481)):
                raise AssertionError('Fixed epoch budget is incomplete')
            selected = min(logs, key=lambda row: (row['validation_nll'], row['epoch']))
            if metadata['selected_epoch'] != selected['epoch'] or metadata['validation_score'] != selected['validation_nll']:
                raise AssertionError('Saved selection is not the earliest all480 validation minimum')
            for filename, key in (('best.pt', 'best_sha256'), ('last.pt', 'last_sha256')):
                if digest(folder / filename) != metadata[key]:
                    raise AssertionError('Checkpoint SHA differs from metadata')
            if metadata['parent_best_sha256'] != digest(parent / 'best.pt') or metadata['parent_last_sha256'] != digest(parent / 'last.pt'):
                raise AssertionError('Continuation parent checkpoint identity changed')
            selections.append({'family': family, 'seed': seed, 'selected_epoch': selected['epoch'],
                               'validation_nll': selected['validation_nll'], 'epochs': len(logs),
                               'original120_curve_identical': True, 'best_sha256': metadata['best_sha256']})
    return len(preserved['files']), selections


def main():
    started, cpu = time.monotonic(), time.process_time()
    preserved_count, selections = verify_preservation_and_selection()
    preparation = json.loads((ROOT / 'results/task03/preparation.json').read_text())
    array_path = ROOT / preparation['array_file']
    if digest(array_path) != preparation['array_sha256']:
        raise AssertionError('Prepared source-array SHA changed')
    manifest = json.loads((ROOT / 'results/task03/split.json').read_text())
    human_groups = {g['index']: g for g in manifest['groups'] if g['split'] == 'test' and g['mechanism'] in MECHANISMS}
    with np.load(array_path, allow_pickle=False) as data:
        offers, next_pool, source_surplus = data['offers'], data['next_pool'], data['surplus']
    eligible = {m: sorted(g for g in human_groups if human_groups[g]['mechanism'] == m and any(v >= 1 for v in offers[g, :, 5])) for m in MECHANISMS}
    if {m: len(ids) for m,ids in eligible.items()} != {'Equal': 5, 'Mixed': 8, 'Proportional': 8}:
        raise AssertionError('Primary eligibility differs from source values')
    scores = list(csv.DictReader((RESULTS / 'forecast_scores.csv').open()))
    selected = {r['cell_id']: r for r in scores if r['origin'] == '5' and r['horizon'] == '10' and r['eligible_at_origin'] == '1'}
    summary = json.loads((RESULTS / 'forecast_summary.json').read_text())
    maxima = defaultdict(float)
    audited_energy = {}
    count = 0
    with sqlite3.connect(f'file:{RESULTS / "forecasts.sqlite3"}?mode=ro', uri=True) as db:
        for identifier, compressed, trajectories in db.execute('SELECT id, body, trajectories FROM cells'):
            if identifier not in selected:
                continue
            row = selected[identifier]
            body = json.loads(zlib.decompress(compressed))
            with np.load(io.BytesIO(trajectories), allow_pickle=False) as paths:
                samples = np.column_stack((paths['pool_after'][:, 9] / 200, paths['window_surplus'][:, 9] / 2000))
            group_index = int(row['group_index'])
            if group_index not in eligible[row['mechanism']]:
                raise AssertionError('Published primary row is not origin-eligible')
            # Derive the target again from untouched source next-pools and
            # recorded rewards, never from an equation-reconstructed future.
            y = np.array([float(next_pool[group_index,14])/200,
                          math.fsum(float(source_surplus[group_index,p,t]) for p in range(4) for t in range(5,15))/2000])
            maxima['observed_endpoint'] = max(maxima['observed_endpoint'], float(np.max(np.abs(y-np.asarray(body['observed_endpoints']['10'])))))
            m = len(samples)
            if m != 64:
                raise AssertionError('Each checkpoint/group bank must have 64 branches')
            exact = math.fsum(math.hypot(*(draw - y)) for draw in samples) / m - math.fsum(pdist(samples)) / (m * (m-1))
            audited_energy[identifier] = exact
            maxima['energy'] = max(maxima['energy'], abs(exact - float(row['energy'])))
            for dimension, label in enumerate(('pool', 'surplus')):
                ordered = sorted(float(v) for v in samples[:, dimension])
                pair_sum = math.fsum((2*i-m+1)*v for i, v in enumerate(ordered))
                proper = math.fsum(abs(v-y[dimension]) for v in ordered) / m - pair_sum/(m*(m-1))
                maxima[f'{label}_crps'] = max(maxima[f'{label}_crps'], abs(proper - float(row[f'{label}_crps'])))
            count += 1
    if count != 1008:
        raise AssertionError(f'Expected 1,008 scored checkpoint/group banks, got {count}')
    effects = []
    for bank, section in (('main', 'primary'), ('second', 'second_bank')):
        for family in ('feedforward', 'recurrent'):
            rows = [r for r in selected.values() if r['bank'] == bank and r['convention'] == 'recorded' and r['family'] == family]
            grouped = defaultdict(dict)
            for row in rows:
                grouped[(row['mechanism'], int(row['group_index']), int(row['seed']))][row['budget']] = audited_energy[row['cell_id']]
            mechanisms = {}
            for mechanism in MECHANISMS:
                groups = sorted({group for m, group, seed in grouped if m == mechanism})
                mechanisms[mechanism] = np.array([math.fsum(grouped[(mechanism, group, seed)]['continued'] - grouped[(mechanism, group, seed)]['original'] for seed in (17, 29, 43)) / 3 for group in groups])
            estimate = math.fsum(math.fsum(v) / len(v) for v in mechanisms.values()) / 3
            random = np.random.Generator(np.random.PCG64(20261014))
            boot = np.zeros(2000)
            for mechanism in MECHANISMS:
                values = mechanisms[mechanism]
                indices = random.integers(len(values), size=(2000, len(values)))
                for i, sample in enumerate(indices):
                    boot[i] += math.fsum(float(values[j]) for j in sample) / len(values) / 3
            interval = np.quantile(boot, [.025, .975])
            saved = next(r for r in summary[section]['comparisons'] if r['family'] == family)
            maxima['effect'] = max(maxima['effect'], abs(estimate-saved['differences']['energy']))
            maxima['interval'] = max(maxima['interval'], float(np.max(np.abs(interval-saved['ci95']['energy']))))
            effects.append({'bank': bank, 'family': family, 'difference': estimate, 'ci95': interval.tolist(), 'groups': sum(len(v) for v in mechanisms.values())})
    if any(value > 2e-12 for value in maxima.values()):
        raise AssertionError(dict(maxima))
    receipt = {'audit_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
               'primary_checkpoint_group_banks': count, 'branches_rechecked': count*64,
               'method': 'Independent SciPy condensed Euclidean distances, sorted scalar CRPS identity and math.fsum group-difference bootstrap; no scoring helper imported',
               'maximum_absolute_disagreement': dict(maxima), 'effects': effects,
               'preserved_files_byte_identical': preserved_count, 'complete_neural_continuations': selections,
               'eligible_counts_from_unchanged_source': {m: len(ids) for m,ids in eligible.items()},
               'bootstrap_replicates': 2000, 'bootstrap_seed': 20261014,
               'forecast_archive_sha256': digest(RESULTS / 'forecasts.sqlite3'),
               'audit_script_sha256': digest(Path(__file__)),
               'wall_seconds': time.monotonic()-started, 'cpu_seconds': time.process_time()-cpu,
               'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    target = RESULTS / 'independent_score_audit.json'
    partial = target.with_suffix('.json.partial')
    partial.write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    partial.replace(target)
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    with (ROOT / 'data/cache/task03/training.lock').open('w') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX | fcntl.LOCK_NB)
        main()
