"""Independent Task 05 saved-evidence audit; no Torch or EvoPolis imports.

Reconstructs proper scores from all archived branches, individual NLL from
saved prequential target log probabilities, and paired human-group intervals.
The raw CSV pass parses numerical fields only for the 32 Experiment 1 test keys.
"""
from collections import defaultdict
import ast
import csv
import datetime
import fcntl
import hashlib
import io
import json
import math
from pathlib import Path
import resource
import sqlite3
import time
import zlib

import numpy as np
from scipy.spatial.distance import pdist

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results/task05'
MECHANISMS = ('Equal', 'Mixed', 'Proportional')
FAMILIES = ('P0', 'P1', 'H0', 'H1')
SEEDS = (17, 29, 43)
CONTRASTS = (('H1', 'H0'), ('P1', 'P0'), ('H1', 'P1'), ('H0', 'P0'))


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def read_csv(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def discrepancy(maxima, name, actual, expected):
    value = float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))
    require(math.isfinite(value), f'Nonfinite audit comparison: {name}')
    maxima[name] = max(maxima[name], value)


def verify_selection():
    preservation = read_json(RESULTS / 'preservation.json')
    for name, expected in preservation['files'].items():
        require(digest(ROOT / name) == expected, f'Previously published artifact changed: {name}')
    frozen = read_json(RESULTS / 'frozen_training.json')['identity']
    require(digest(ROOT / 'configs/task05.json') == frozen['config_sha256'], 'Frozen configuration changed')
    require(digest(ROOT / 'results/task03/split.json') == frozen['split_sha256'], 'Frozen split changed')
    for name, expected in frozen['code_sha256'].items():
        require(digest(ROOT / name) == expected, f'Frozen training source changed: {name}')
    correction = verify_correction(frozen)
    selected = {}
    for family in FAMILIES:
        for seed in SEEDS:
            folder = RESULTS / 'weights' / f'{family}_{seed}'
            logs, metadata = read_json(folder / 'training.json'), read_json(folder / 'metadata.json')
            require([r['epoch'] for r in logs] == list(range(1, 481)), f'Incomplete epoch budget: {family}/{seed}')
            best = min(logs, key=lambda row: (row['validation_nll'], row['epoch']))
            require(metadata['selected_epoch'] == best['epoch'] and metadata['validation_score'] == best['validation_nll'],
                    f'Selection is not earliest validation minimum: {family}/{seed}')
            for filename, field in (('best.pt', 'best_sha256'), ('last.pt', 'last_sha256')):
                require(digest(folder / filename) == metadata[field], f'Checkpoint hash changed: {family}/{seed}/{filename}')
            for name in ('beta', 'eta', 'sigma'):
                require(metadata[name] == best[name], f'Selected parameter mismatch: {family}/{seed}/{name}')
            expected_nodes = frozen['configuration']['quadrature_nodes']
            if correction and family.startswith('P'):
                original = correction['unaffected_fits'][f'{family}_{seed}']
                require(metadata['best_sha256'] == original['best_sha256'] and metadata['last_sha256'] == original['last_sha256'],
                        f'An unaffected population fit changed during the numerical correction: {family}/{seed}')
                expected_nodes = frozen['configuration']['quadrature_initial_nodes']
            require(metadata['integration_nodes'] == expected_nodes, f'Wrong recorded integration resolution: {family}/{seed}')
            selected[family, seed] = {'family': family, 'seed': seed, 'epochs': 480,
                                     'selected_epoch': best['epoch'], 'validation_nll': best['validation_nll'],
                                     'best_sha256': metadata['best_sha256'], 'integration_nodes': metadata['integration_nodes']}
    integration = read_json(RESULTS / 'integration_checks.json')
    require(integration['passed'] and integration['all_six_persistent_fits_checked'], 'Final integration gate is incomplete')
    checks = {(row['family'], row['seed']): row for row in integration['checks']}
    require(len(checks) == 6, 'Missing or duplicate final integration checks')
    for family in ('H0', 'H1'):
        for seed in SEEDS:
            check, fit = checks[family, seed], selected[family, seed]
            require(check['checkpoint_sha256'] == fit['best_sha256'], 'Integration gate checked a different checkpoint')
            require(check['nodes_compared'][0] == fit['integration_nodes'], 'Integration gate used a different prediction resolution')
            require(check['passed'] and all(check['maxima'][name] < .001 for name in ('group_nll', 'resident_nll', 'prediction_fraction', 'tail_probability')),
                    'Integration accuracy threshold failed')
            require(check['maxima']['posterior_normalization'] < 1e-12, 'Posterior normalization threshold failed')
    opening = read_json(RESULTS / 'evaluation_opening.json')
    require(opening['contract']['integration_sha256'] == digest(RESULTS / 'integration_checks.json'), 'Pre-test integration evidence changed')
    require(datetime.datetime.fromisoformat(opening['opened_utc']) >= datetime.datetime.fromisoformat(integration['utc']), 'Evaluation opened before the final numerical gate')
    correction_summary = None if correction is None else {
        'receipt': 'results/task05/numerical_correction_01/receipt.json',
        'receipt_sha256': digest(RESULTS / 'numerical_correction_01/receipt.json'),
        'selected_integration_nodes': correction['selected_integration_nodes'],
        'original_sources_hash_verified': len(correction['original_files']),
        'unaffected_P_checkpoints_byte_identical': len(correction['unaffected_fits']),
        'persistent_fits_refit': correction['refit_from_seeded_initialization'],
        'training_model_functions_ast_identical': True,
        'all_six_final_integration_checks_passed': True}
    return len(preservation['files']), selected, correction_summary


def verify_correction(frozen):
    """Preserve documented original evidence while checking the corrected freeze."""
    if not frozen['configuration'].get('numerical_correction'):
        return None
    folder = RESULTS / 'numerical_correction_01'
    receipt = read_json(folder / 'receipt.json')
    require(receipt['test_outcome_comparisons_opened'] is False, 'Numerical correction was not declared before new outcome comparisons')
    require(receipt['selected_integration_nodes'] == frozen['configuration']['quadrature_nodes'], 'Corrected resolution differs from configuration')
    for original, expected in receipt['original_files'].items():
        require(digest(folder / Path(original).name) == expected, f'Original correction evidence changed: {original}')
    for family_seed, attempt in receipt['original_attempts'].items():
        require(digest(folder / 'weights' / family_seed / 'last.pt') == attempt['last_sha256'], f'Original attempted fit changed: {family_seed}')
    before = ast.parse((folder / 'conditional_models.py').read_text())
    after = ast.parse((ROOT / 'evopolis/conditional_models.py').read_text())
    definitions = lambda tree: {node.name: ast.dump(node, include_attributes=False) for node in tree.body
                               if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name != 'prequential'}
    require(definitions(before) == definitions(after), 'Training model definitions changed during the quadrature correction')
    equivalence = read_json(folder / 'inference_equivalence.json')
    require(equivalence['passed'] and equivalence['before_sha256'] == digest(folder / 'conditional_models.py')
            and equivalence['after_sha256'] == digest(ROOT / 'evopolis/conditional_models.py'), 'Inference-equivalence evidence does not match the model code')
    require(all(value < 1e-12 for value in equivalence['max_abs_disagreements'].values()), 'Inference optimization equivalence failed')
    return receipt


def load_source(maxima):
    manifest = read_json(ROOT / 'results/task03/split.json')
    preparation = read_json(ROOT / 'results/task03/preparation.json')
    path = ROOT / preparation['array_file']
    require(digest(path) == preparation['array_sha256'], 'Prepared source-array hash changed')
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in ('offers', 'surplus', 'next_pool', 'pool', 'y')}
    groups = {g['index']: g for g in manifest['groups'] if g['split'] == 'test'}
    require(len(groups) == 32, 'Original 32-group test split changed')
    keys = {tuple(g['key']): g for g in groups.values()}
    allowed_conditions = {'Equal Baseline Exp 1', 'Mixed Baseline Exp 1', 'Proportional Baseline Exp 1', 'RL Agent (M1) Exp 1'}
    require(all(key[0] in allowed_conditions for key in keys), 'Unexpected test condition')
    source = ROOT / 'data/raw' / manifest['source_file']
    require(digest(source) == manifest['source_sha256'], 'Pinned raw human source changed')
    observed = defaultdict(dict)
    with source.open(newline='') as stream:
        reader = csv.reader(stream)
        header = next(reader)
        column = {name: i for i, name in enumerate(header)}
        for row in reader:
            # Never parse outcomes of reserved experiments or upstream models.
            condition = row[column['mech_name_by_player']]
            if condition not in allowed_conditions:
                continue
            key = (condition, row[column['launch_id']], row[column['episode_id']])
            if key not in keys:
                continue
            t = int(row[column['round_id']])
            require(t not in observed[key] and 0 <= t < 40, 'Duplicate or invalid source round')
            observed[key][t] = {
                'offers': [float(row[column[f'offer_{p}']]) for p in range(4)],
                'y': [float(row[column[f'player_action_{p}']]) for p in range(4)],
                'surplus': [float(row[column[f'player_reward_{p}']]) for p in range(4)],
                'pool': float(row[column['mechanism_observation.pool']]),
                'next_pool': float(row[column['next_environment_state.pool']])}
    for key, group in keys.items():
        require(sorted(observed[key]) == list(range(40)), f'Incomplete raw source group: {key}')
        index = group['index']
        for t, record in observed[key].items():
            for name in ('offers', 'y', 'surplus'):
                discrepancy(maxima, 'raw_source_arrays', record[name], arrays[name][index, :, t])
            for name in ('pool', 'next_pool'):
                discrepancy(maxima, 'raw_source_arrays', record[name], arrays[name][index, t])
    eligible = {m: sorted((tuple(g['key']) for g in groups.values() if g['mechanism'] == m
                          and any(e >= 1 for e in observed[tuple(g['key'])][5]['offers']))) for m in MECHANISMS}
    require({m: len(v) for m, v in eligible.items()} == {'Equal': 5, 'Mixed': 8, 'Proportional': 8}, 'Primary eligibility changed')
    previous = read_json(ROOT / 'results/task04/numerical_audit.json')
    old_keys = sorted(tuple(r['key']) for r in previous['origins'] if r['origin'] == 5 and r['eligible'])
    require(sorted(key for keys_in_mechanism in eligible.values() for key in keys_in_mechanism) == old_keys, 'Task 04 primary keys changed')
    return arrays, groups, observed, eligible, manifest['source_sha256']


def scalar_crps(draws, target):
    ordered = sorted(float(v) for v in draws)
    count = len(ordered)
    # Sum_{i<j}|x_i-x_j| through order-statistic coefficients.
    pair_sum = math.fsum((2*i-count+1)*value for i, value in enumerate(ordered))
    return math.fsum(abs(value-target) for value in ordered)/count - pair_sum/(count*(count-1))


def audit_forecasts(groups, observed, selections, maxima):
    published = read_csv(RESULTS / 'forecast_scores.csv')
    rows = {(r['cell_id'], int(r['horizon'])): r for r in published}
    require(len(rows) == len(published) == 5364, 'Forecast score rows are missing or duplicated')
    audited = {}
    banks = defaultdict(int)
    scores_checked = 0
    with sqlite3.connect((RESULTS / 'forecasts.sqlite3').resolve().as_uri() + '?mode=ro', uri=True) as database:
        for identifier, metadata_json, body_blob, trajectory_blob in database.execute('SELECT id,metadata,body,trajectories FROM cells ORDER BY id'):
            metadata = json.loads(metadata_json)
            body = json.loads(zlib.decompress(body_blob))
            with np.load(io.BytesIO(trajectory_blob), allow_pickle=False) as archive:
                pool, surplus = archive['pool_after'], archive['window_surplus']
            key = tuple(metadata['key'])
            group = groups[int(metadata['group_index'])]
            require(key == tuple(group['key']), 'Archive group index and complete source key disagree')
            family, seed = metadata['family'], int(metadata['training_seed'])
            selected = selections[family, seed]
            require(metadata['checkpoint_sha256'] == selected['best_sha256'] and metadata['selected_epoch'] == selected['selected_epoch'], 'Archive selected checkpoint changed')
            origin, horizon = int(metadata['origin']), int(metadata['horizon'])
            require(pool.shape == surplus.shape == (64, horizon), 'Incorrect archived branch shape')
            require(metadata['bank'] in ('main', 'second'), 'Unexpected forecast bank')
            require(horizon == (20 if metadata['bank'] == 'main' else 10), 'Incorrect bank horizon')
            banks[metadata['bank']] += 1
            path_scores = [scalar_crps(pool[:, offset]/200, observed[key][origin+offset]['next_pool']/200) for offset in range(horizon)]
            for h in (1, 5, 10, 20):
                if h > horizon:
                    continue
                row = rows[identifier, h]
                require(tuple(json.loads(row['group_key'])) == key and row['mechanism'] == group['mechanism'], 'Score full group key changed')
                require(int(row['seed']) == seed and row['family'] == family, 'Score model identity changed')
                require(int(row['group_index']) == group['index'] and int(row['origin']) == origin and row['bank'] == metadata['bank'], 'Score origin/bank identity changed')
                require(row['checkpoint_sha256'] == selected['best_sha256'], 'Score checkpoint hash changed')
                target = np.array([observed[key][origin+h-1]['next_pool']/200,
                                   math.fsum(observed[key][t]['surplus'][p] for p in range(4) for t in range(origin, origin+h))/(200*h)])
                samples = np.column_stack((pool[:, h-1]/200, surplus[:, h-1]/(200*h)))
                discrepancy(maxima, 'observed_endpoint', target, body['observed_endpoints'][str(h)])
                discrepancy(maxima, 'saved_endpoints', samples, body['endpoints'][str(h)])
                energy = math.fsum(math.hypot(*(draw-target)) for draw in samples)/64 - math.fsum(pdist(samples))/(64*63)
                discrepancy(maxima, 'energy', energy, float(row['energy']))
                discrepancy(maxima, 'trajectory_pool_crps', math.fsum(path_scores[:h])/h, float(row['trajectory_pool_crps']))
                for dimension, name in enumerate(('pool', 'surplus')):
                    discrepancy(maxima, name+'_crps', scalar_crps(samples[:, dimension], target[dimension]), float(row[name+'_crps']))
                    low, high = np.quantile(samples[:, dimension], (.1, .9), method='linear')
                    discrepancy(maxima, 'interval_bounds', [low, high], [float(row[name+'_lower80']), float(row[name+'_upper80'])])
                    discrepancy(maxima, 'coverage', int(low <= target[dimension] <= high), float(row[name+'_coverage80']))
                    discrepancy(maxima, 'forecast_means', math.fsum(samples[:, dimension])/64, float(row[name+'_mean']))
                    discrepancy(maxima, 'reported_target', target[dimension], float(row['observed_'+name+'_scaled']))
                eligible = any(e >= 1 for e in observed[key][origin]['offers'])
                require(int(row['eligible_at_origin']) == int(eligible), 'Score origin eligibility differs')
                if origin == 5 and h == 10 and eligible:
                    identity = (metadata['bank'], family, seed, key)
                    require(identity not in audited, 'Duplicate primary checkpoint/group bank')
                    audited[identity] = energy
                scores_checked += 1
    require(dict(banks) == {'main': 1152, 'second': 252}, f'Incomplete banks: {dict(banks)}')
    require(scores_checked == len(rows), 'Unaudited score rows remain')
    require(len(audited) == 504, 'Incomplete primary checkpoint/group banks')
    return audited, {'cells': sum(banks.values()), 'branches': sum(banks.values())*64, 'score_rows': scores_checked,
                     'primary_checkpoint_group_banks': len(audited), 'bank_cells': dict(banks)}


def audit_individual(arrays, groups, eligible, source_sha256, maxima):
    identity = read_json(RESULTS / 'test_prequential_identity.json')
    require(identity['source_sha256'] == source_sha256, 'Prequential source hash changed')
    require(identity['split_sha256'] == digest(ROOT / 'results/task03/split.json'), 'Prequential split hash changed')
    require(identity['array_sha256'] == digest(RESULTS / 'test_prequential_log_probs.npz'), 'Prequential log-probability hash changed')
    with np.load(RESULTS / 'test_prequential_log_probs.npz', allow_pickle=False) as archive:
        probabilities = archive['log_prob']
    specs, ordered_groups = identity['record_order'], identity['groups']
    require(probabilities.shape == (24, 32, 40, 4), 'Incomplete saved prequential log probabilities')
    primary_keys = {key for keys in eligible.values() for key in keys}
    primary_rows = {(r['family'], r['budget'], int(r['seed']), tuple(json.loads(r['group_key']))): r
                    for r in read_csv(RESULTS / 'individual_primary_window.csv')}
    full_rows = {(r['family'], r['budget'], int(r['seed']), tuple(json.loads(r['group_key']))): r
                 for r in read_csv(RESULTS / 'individual_groups.csv')}
    require(len(primary_rows) == 504 and len(full_rows) == 768, 'Incomplete individual score rows')
    audited = {}
    for spec_index, spec in enumerate(specs):
        for position, group in enumerate(ordered_groups):
            index, key = int(group['index']), tuple(group['key'])
            require(key == tuple(groups[index]['key']), 'Prequential array full group identity changed')
            mask = arrays['offers'][index].T >= 1
            log_prob = probabilities[spec_index, position]
            require(np.isfinite(log_prob).all(), 'Nonfinite saved target log probability')
            require(np.max(np.abs(log_prob[~mask]), initial=0.) < 1e-12, 'Forced choice has nonunit probability')
            family, budget, seed = spec['family'], spec['budget'], int(spec['seed'])
            row_key = family, budget, seed, key
            for label, time_slice, table in (('full', slice(0, 40), full_rows), ('primary', slice(5, 15), primary_rows)):
                if label == 'primary' and key not in primary_keys:
                    continue
                selected = log_prob[time_slice][mask[time_slice]]
                require(len(selected) > 0, 'Primary individual group has no nonforced choices')
                nll = -math.fsum(float(value) for value in selected)/len(selected)
                row = table[row_key]
                require(int(row['nonforced_choices']) == len(selected), 'Individual score choice count changed')
                require(int(row['group_index']) == index and row['mechanism'] == group['mechanism'], 'Individual group identity changed')
                discrepancy(maxima, 'individual_nll' if family in FAMILIES else 'reference_individual_nll', nll, float(row['nll']))
                if label == 'primary' and family in FAMILIES:
                    audited[family, seed, key] = nll
    require(len(audited) == 252, 'Incomplete conditional-model primary individual scores')
    return audited


def audit_bootstrap(energy, nll, eligible, maxima):
    summary = read_json(RESULTS / 'forecast_summary.json')
    rng = np.random.Generator(np.random.PCG64(20261022))
    resampling = {m: rng.integers(len(eligible[m]), size=(2000, len(eligible[m]))) for m in MECHANISMS}
    effects = []
    for bank, section in (('main', 'primary'), ('second', 'second_bank')):
        saved = summary[section]
        require(saved['bootstrap']['seed'] == 20261022 and saved['bootstrap']['replicates'] == 2000, 'Bootstrap plan changed')
        for mechanism in MECHANISMS:
            saved_keys = [tuple(json.loads(value)) if isinstance(value, str) else tuple(value) for value in saved['bootstrap']['group_keys'][mechanism]]
            require(saved_keys == eligible[mechanism], 'Bootstrap ordering differs from full-key order')
        for new, reference in CONTRASTS:
            published = next(row for row in saved['comparisons'] if row['comparison'] == f'{new} minus {reference}')
            result = {'bank': bank, 'comparison': f'{new} minus {reference}', 'groups': 21, 'differences': {}, 'ci95': {}}
            for metric, source in (('nll', nll), ('energy', energy)):
                values = {}
                for mechanism in MECHANISMS:
                    def get(family, seed, key):
                        return source[(bank, family, seed, key) if metric == 'energy' else (family, seed, key)]
                    values[mechanism] = [math.fsum(get(new, seed, key)-get(reference, seed, key) for seed in SEEDS)/3 for key in eligible[mechanism]]
                estimate = math.fsum(math.fsum(values[m])/len(values[m]) for m in MECHANISMS)/3
                boot = np.array([math.fsum(math.fsum(values[m][int(j)] for j in resampling[m][replicate])/len(values[m]) for m in MECHANISMS)/3 for replicate in range(2000)])
                interval = np.quantile(boot, (.025, .975))
                discrepancy(maxima, 'contrast_estimate', estimate, published['differences'][metric])
                discrepancy(maxima, 'contrast_interval', interval, published['ci95'][metric])
                result['differences'][metric], result['ci95'][metric] = estimate, interval.tolist()
            effects.append(result)
    primary = next(row for row in effects if row['bank'] == 'main' and row['comparison'] == 'H1 minus H0')
    joint = all(primary['ci95'][metric][1] < 0 for metric in ('nll', 'energy'))
    require(summary['joint_predictive_improvement'] == joint, 'Joint-improvement declaration differs from intersection rule')
    return effects, joint


def main():
    started, cpu = time.monotonic(), time.process_time()
    maxima = defaultdict(float)
    preserved, selections, correction = verify_selection()
    arrays, groups, observed, eligible, source_sha256 = load_source(maxima)
    energy, forecast_counts = audit_forecasts(groups, observed, selections, maxima)
    nll = audit_individual(arrays, groups, eligible, source_sha256, maxima)
    effects, joint = audit_bootstrap(energy, nll, eligible, maxima)
    for name, value in maxima.items():
        require(value < (1e-6 if name == 'reference_individual_nll' else 2e-12), f'Independent audit disagreement: {name}={value}')
    receipt = {'audit_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'passed': True,
               'method': 'No Torch or EvoPolis imports. SciPy condensed Euclidean distances, sorted scalar CRPS, math.fsum target log probabilities and paired full-key group-difference bootstrap.',
               'preserved_files_byte_identical': preserved, 'complete_fits': list(selections.values()),
               'documented_numerical_correction': correction,
               'forecast_audit': forecast_counts, 'raw_source_groups': 32, 'raw_source_rounds': 1280,
               'raw_source_sha256': source_sha256, 'reserved_experiment_outcomes_parsed': False,
               'primary_full_keys_by_mechanism': eligible, 'bootstrap_seed': 20261022, 'bootstrap_replicates': 2000,
               'effects': effects, 'joint_predictive_improvement': joint,
               'maximum_absolute_disagreement': dict(maxima),
               'forecast_archive_sha256': digest(RESULTS / 'forecasts.sqlite3'),
               'audit_script_sha256': digest(Path(__file__)), 'wall_seconds': time.monotonic()-started,
               'cpu_seconds': time.process_time()-cpu, 'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    target = RESULTS / 'independent_score_audit.json'
    temporary = target.with_suffix('.json.partial')
    temporary.write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    temporary.replace(target)
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    with (ROOT / 'data/cache/task03/training.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        main()
