"""Task 05 past-only conditional forecasts and reproducible frozen banks.

Task 04's archive format, proper-score targets and pure resource helpers are
reused unchanged. Latent draws and action uniforms have separate explicit
indices; every generated branch retains one resident effect throughout play.
"""
import argparse
from contextlib import closing
import datetime
import fcntl
import hashlib
import json
import math
from pathlib import Path
import resource
import sqlite3
import time
import zlib

import numpy as np

from .behavior_data import ROOT, content_hash, load, write_json
from .forecast_generate import (baseline_groups, eligible, observed_rounds, packed,
                                pack_paths, unpack_paths, numerical_audit)
from .sources import sha256
from .world import CAPACITY, PLAYERS, ROUNDS, WorldState, allocate, step

RESULTS = ROOT / 'results/task05'
ARCHIVE = RESULTS / 'forecasts.sqlite3'
CONFIG = ROOT / 'configs/task05.json'
FAMILIES = ('P0', 'P1', 'H0', 'H1')
SEEDS = (17, 29, 43)
ORIGINS = (0, 5, 10, 20)
HORIZONS = (1, 5, 10, 20)
NAMESPACE = 20261023


def checkpoint_registry():
    configured_nodes = int(json.loads(CONFIG.read_text())['quadrature_nodes'])
    rows = []
    for family in FAMILIES:
        for seed in SEEDS:
            folder = RESULTS / 'weights' / f'{family}_{seed}'
            metadata = json.loads((folder / 'metadata.json').read_text())
            if metadata['epochs'] != 480:
                raise ValueError('A selected model has not completed the full epoch budget')
            if family.startswith('H') and int(metadata['integration_nodes']) != configured_nodes:
                raise ValueError('Persistent checkpoint integration differs from the frozen corrected configuration')
            path = folder / 'best.pt'
            digest = sha256(path)
            if metadata.get('best_sha256', digest) != digest:
                raise ValueError('Selected checkpoint differs from metadata')
            rows.append({'family': family, 'seed': seed, 'training_seed': seed,
                         'budget': 'conditional', 'budget_epochs': 480,
                         'path': str(path.relative_to(ROOT)), 'sha256': digest,
                         'checkpoint_sha256': digest, 'selected_epoch': metadata['selected_epoch'],
                         'integration_nodes': int(metadata['integration_nodes'])})
    return rows


def load_checkpoint(record):
    import torch
    from .conditional_models import ConditionalModel
    path = ROOT / record['path']
    if sha256(path) != record['sha256']:
        raise ValueError('Frozen selected checkpoint changed')
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if (checkpoint['family'], checkpoint['seed'], checkpoint['epoch']) != (record['family'], record['seed'], record['selected_epoch']):
        raise ValueError('Selected checkpoint identity differs')
    if int(checkpoint['integration_nodes']) != record['integration_nodes']:
        raise ValueError('Selected checkpoint integration differs from its metadata')
    model = ConditionalModel(record['family'])
    model.load_state_dict(checkpoint['model_state'])
    return model.eval(), checkpoint



def validate_integration(registry, report=None):
    """Require the development accuracy check of these exact persistent fits."""
    if report is None:
        report = json.loads((RESULTS / 'integration_checks.json').read_text())
    if not report['passed'] or not report.get('all_six_persistent_fits_checked'):
        raise ValueError('All six persistent fits must pass integration before test scoring or forecasting')
    checks = {(row['family'], int(row['seed'])): row for row in report['checks']}
    expected = {(r['family'], r['seed']) for r in registry if r['family'].startswith('H')}
    if len(checks) != len(report['checks']) or set(checks) != expected:
        raise ValueError('Integration receipt has missing or duplicate persistent checkpoints')
    for record in registry:
        if not record['family'].startswith('H'):
            continue
        check = checks[(record['family'], record['seed'])]
        if not check['passed'] or check['checkpoint_sha256'] != record['sha256']:
            raise ValueError('Integration receipt does not certify the selected checkpoint')
        compared = check['nodes_compared']
        if compared[0] != record['integration_nodes'] or compared[1] <= compared[0]:
            raise ValueError('Integration receipt uses a different prediction resolution')
    return report


def seed_table(arrays, manifest):
    rows = []
    for group in baseline_groups(manifest):
        # Full human key enters the namespace, not a possibly reordered row id.
        digest = hashlib.sha256(packed(group['key']).encode()).digest()
        key_words = np.frombuffer(digest, dtype='<u4').astype(np.uint32).tolist()
        for origin in ORIGINS:
            for family_index, family in enumerate(FAMILIES):
                for seed in SEEDS:
                    for bank_index, bank in enumerate(('main', 'second')):
                        if bank == 'second' and (origin != 5 or not eligible(arrays, group, origin)):
                            continue
                        seeds = [int(np.random.SeedSequence([NAMESPACE, *key_words, origin, family_index, seed, bank_index, branch]).generate_state(1, dtype=np.uint64)[0]) for branch in range(64)]
                        rows.append({'group_index': group['index'], 'key': group['key'], 'origin': origin,
                                     'family': family, 'training_seed': seed, 'bank': bank,
                                     'rollout_seeds': [str(s) for s in seeds]})
    flat = [s for row in rows for s in row['rollout_seeds']]
    if len(flat) != 89856 or len(set(flat)) != 89856:
        raise ValueError('Incomplete or colliding branch seed plan')
    return {'namespace': NAMESPACE, 'rows': rows, 'table_hash': content_hash(rows),
            'distinct_branch_streams': len(flat), 'numpy_version': np.__version__,
            'algorithm': 'SeedSequence([20261023, eight little-endian uint32 words of SHA256(compact JSON full human key), origin, family index P0/P1/H0/H1, training seed, bank index main/second, branch]).generate_state(1,uint64)',
            'resident_streams': 'PCG64(SeedSequence([branch_seed,resident,purpose])); purpose 0 = one latent inverse-CDF uniform; purpose 1 = array of horizon action inverse-CDF uniforms indexed by offset',
            'coupling': 'Independent family-keyed streams; no common random numbers across families. Draw consumption never shifts later round uniforms.'}


def prepare():
    arrays, manifest = load()
    audit = numerical_audit(arrays, manifest)
    primary = [r for r in audit['origins'] if r['origin'] == 5 and r['eligible']]
    old = json.loads((ROOT / 'results/task04/numerical_audit.json').read_text())
    old_keys = sorted(r['key'] for r in old['origins'] if r['origin'] == 5 and r['eligible'])
    if sorted(r['key'] for r in primary) != old_keys:
        raise ValueError('Task 04 primary full group keys changed')
    audit['primary_full_group_keys'] = old_keys
    table = seed_table(arrays, manifest)
    for name, value in (('numerical_audit.json', audit), ('forecast_seeds.json', table)):
        path = RESULTS / name
        if path.exists() and json.loads(path.read_text()) != value:
            raise ValueError(f'Refusing to replace frozen {name}')
        write_json(path, value)
    return {'primary_groups': len(primary), 'branches': len(table['rows']) * 64}


def indexed_uniforms(seeds, horizon):
    latent = np.empty((len(seeds), 4))
    actions = np.empty((len(seeds), horizon, 4))
    for branch, seed in enumerate(seeds):
        for resident in range(4):
            latent[branch, resident] = np.random.Generator(np.random.PCG64(np.random.SeedSequence([int(seed), resident, 0]))).random()
            actions[branch, :, resident] = np.random.Generator(np.random.PCG64(np.random.SeedSequence([int(seed), resident, 1]))).random(horizon)
    return latent, actions


def inverse_cdf(probabilities, uniform):
    """One fixed uniform selects an exact legal action, including point masses."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if np.any(probabilities < 0) or not np.isclose(probabilities.sum(), 1, atol=1e-10, rtol=1e-10):
        raise ValueError('Unnormalized sampling probabilities')
    if not 0 <= uniform < 1:
        raise ValueError('Inverse-CDF uniform must lie in [0,1)')
    cumulative = np.cumsum(probabilities)
    # Setting the mathematically unit endpoint to one does not change the PMF.
    cumulative[-1] = 1.0
    return int(np.searchsorted(cumulative, uniform, side='right'))


def posterior_summary(nodes, weights):
    result = []
    for resident in range(4):
        p = weights[resident]
        mean = float(np.dot(p, nodes))
        result.append({'mean': mean, 'sd': float(np.sqrt(np.dot(p, (nodes - mean)**2))),
                       'p10': float(nodes[inverse_cdf(p, .1)]), 'p50': float(nodes[inverse_cdf(p, .5)]),
                       'p90': float(nodes[inverse_cdf(p, .9)]),
                       'effective_nodes': float(1 / np.dot(p, p)),
                       'normalization': float(p.sum())})
    return result


def simulate_branches(model, arrays, group, origin, seeds, *, horizon=20, nodes=None):
    import torch
    from .conditional_models import infer_prefix, update_history
    if group['mechanism'] not in ('Equal', 'Mixed', 'Proportional'):
        raise ValueError('No executable live M1 or Interpolating forecast')
    if origin < 0 or horizon < 1 or origin+horizon > ROUNDS or not seeds:
        raise ValueError('Forecast must lie inside the recorded horizon')
    index, count = group['index'], len(seeds)
    if nodes is None:
        nodes = int(json.loads(CONFIG.read_text())['quadrature_nodes'])
    with torch.no_grad():
        prefix = infer_prefix(model, arrays, index, origin, nodes=nodes)
    latent_nodes = prefix['nodes'].detach().numpy()
    weights = prefix['weights'].detach().numpy()
    latent_uniforms, action_uniforms = indexed_uniforms(seeds, horizon)
    sampled_u = np.zeros((count, 4))
    for branch in range(count):
        for resident in range(4):
            sampled_u[branch, resident] = latent_nodes[inverse_cdf(weights[resident], latent_uniforms[branch, resident])]
    history = {key: value.unsqueeze(0).repeat(count, 1) for key, value in prefix['history'].items()}
    pool = float(arrays['pool'][index, origin])
    first_offers = tuple(arrays['offers'][index, :, origin].tolist())
    if min(first_offers) < 0 or math.fsum(first_offers) > pool:
        raise ValueError('Infeasible recorded origin')
    previous = None if origin == 0 else tuple(arrays['y'][index, :, origin-1].tolist())
    states = [WorldState(pool, origin, previous) for _ in seeds]
    paths = {'pool_before': np.zeros((count, horizon)), 'pool_after': np.zeros((count, horizon)),
             'window_surplus': np.zeros((count, horizon)), 'participation': np.zeros((count, horizon), dtype=np.int8),
             'offers': np.zeros((count, horizon, 4)), 'contributions': np.zeros((count, horizon, 4), dtype=np.int16),
             'surplus': np.zeros((count, horizon, 4)), 'pmfs': np.zeros((count, horizon, 4, 201)),
             'padded': np.ones((count, horizon), dtype=bool), 'actual_rounds': np.zeros(count, dtype=np.int16),
             'sampled_u': sampled_u,
             **{f'history_{key}': np.zeros((count, horizon, 4)) for key in history}}
    paths['pmfs'][..., 0] = 1
    for offset in range(horizon):
        if offset:
            paths['window_surplus'][:, offset] = paths['window_surplus'][:, offset-1]
        # Histories persist even after exact-zero termination; no observations invented.
        for key, value in history.items():
            paths[f'history_{key}'][:, offset] = value.numpy()
        live = [i for i, state in enumerate(states) if not state.done]
        if not live:
            continue
        offers = [first_offers if offset == 0 else allocate(states[i].pool, states[i].previous_contributions, mechanism=group['mechanism'].lower()) for i in live]
        offer_tensor = torch.tensor(offers, dtype=torch.float64)
        selected_history = {key: value[live] for key, value in history.items()}
        with torch.no_grad():
            pmfs = model.probs(torch.tensor([states[i].pool for i in live], dtype=torch.float64), offer_tensor,
                               selected_history, torch.tensor(sampled_u[live], dtype=torch.float64)).numpy()
        actions = [[inverse_cdf(pmfs[j, resident, :math.floor(offers[j][resident])+1], action_uniforms[i, offset, resident]) for resident in range(4)] for j, i in enumerate(live)]
        # All four PMFs and choices precede any history update.
        with torch.no_grad():
            updated = update_history(selected_history, offer_tensor, torch.tensor(actions, dtype=torch.float64), model.eta)
        for key in history:
            history[key][live] = updated[key]
        for j, i in enumerate(live):
            prior = states[i]
            states[i], outcome = prior.advance(offers[j], actions[j], integer_contributions=True)
            paths['pool_before'][i, offset] = prior.pool
            paths['pool_after'][i, offset] = outcome.next_pool
            paths['offers'][i, offset] = offers[j]
            paths['contributions'][i, offset] = actions[j]
            paths['surplus'][i, offset] = outcome.surplus
            paths['window_surplus'][i, offset] += math.fsum(outcome.surplus)
            paths['participation'][i, offset] = sum(e >= 1 for e in offers[j])
            paths['pmfs'][i, offset] = pmfs[j]
            paths['padded'][i, offset] = False
            paths['actual_rounds'][i] += 1
    if not all(np.isfinite(value).all() for value in paths.values()):
        raise FloatingPointError('Nonfinite generated trajectory')
    details = {'latent_effects': sampled_u, 'prefix_posteriors': posterior_summary(latent_nodes, weights),
               'prefix_posterior_nodes': latent_nodes.tolist(), 'prefix_posterior_weights': weights.tolist(),
               'parameters': {name: float(getattr(model, name).detach()) for name in ('beta', 'eta', 'sigma')}}
    return paths, details


def branch_episode(paths, details, branch_index, arrays, group, origin, seeds):
    prefix = [math.fsum(arrays['surplus'][group['index'], p, :origin]) for p in range(4)]
    rows = []
    horizon = paths['pool_after'].shape[1]
    history_names = [name[8:] for name in paths if name.startswith('history_')]
    for offset in range(horizon):
        offers = paths['offers'][branch_index, offset].tolist()
        returns = paths['contributions'][branch_index, offset].tolist()
        cumulative = [prefix[p] + math.fsum(paths['surplus'][branch_index, :offset+1, p]) for p in range(4)]
        padded = bool(paths['padded'][branch_index, offset])
        predictions = []
        for p in range(4):
            n = math.floor(offers[p])
            pmf = paths['pmfs'][branch_index, offset, p, :n+1]
            predictions.append({'expected_contribution': float(np.dot(np.arange(n+1), pmf)),
                                'p_zero': float(pmf[0]), 'p_max': float(pmf[-1]), 'legal_max': n,
                                'evaluated': not padded, 'pmf': pmf.tolist(),
                                'sampled_latent_effect': float(paths['sampled_u'][branch_index, p]),
                                'latent_conditioning': 'sampled persistent resident effect' if details['parameters']['sigma'] > 0 else 'u=0',
                                'history': {name: float(paths['history_'+name][branch_index, offset, p]) for name in history_names}})
        after = float(paths['pool_after'][branch_index, offset])
        rows.append({'round_id': origin+offset, 'pool_before': float(paths['pool_before'][branch_index, offset]), 'pool_after': after,
                     'offers': offers, 'contributions': returns, 'surplus': paths['surplus'][branch_index, offset].tolist(),
                     'cumulative_surplus': cumulative, 'group_cumulative_surplus': math.fsum(cumulative),
                     'window_cumulative_surplus': float(paths['window_surplus'][branch_index, offset]),
                     'equation_after': after, 'pool_residual': 0., 'equation_input_valid': True,
                     'active_count': int(paths['participation'][branch_index, offset]), 'padded': padded, 'observed': False,
                     'after_source': 'post-termination zero padding' if padded else 'conditional model forecast through WorldState',
                     'predictions': predictions})
    return {'branch_index': branch_index, 'rollout_seed': str(seeds[branch_index]), 'rounds': rows,
            'actual_rounds': int(paths['actual_rounds'][branch_index]), 'exact_zero_termination': bool(paths['pool_after'][branch_index, -1] == 0),
            'latent_effects': paths['sampled_u'][branch_index].tolist(), 'prefix_posteriors': details['prefix_posteriors'],
            'parameters': details['parameters']}


def generate_cell(model, arrays, group, origin, seeds, *, horizon=20, nodes=None):
    paths, details = simulate_branches(model, arrays, group, origin, seeds, horizon=horizon, nodes=nodes)
    executed = np.argwhere(~paths['padded'])
    residual = max((abs(step(paths['pool_before'][i, t], paths['offers'][i, t], paths['contributions'][i, t], integer_contributions=True).next_pool-paths['pool_after'][i, t]) for i, t in executed), default=0.)
    legal = max(float(np.max(-paths['contributions'])), float(np.max(paths['contributions']-np.floor(paths['offers']))))
    overshoot = max((math.fsum(paths['offers'][i, t])-paths['pool_before'][i, t] for i, t in executed), default=0.)
    if residual != 0 or legal > 0 or overshoot > 0:
        raise ValueError('Generated resource/support audit failed')
    audit = {'executed_rounds': len(executed), 'padded_rounds': int(paths['padded'].sum()),
             'max_abs_transition_residual': residual, 'max_legal_support_violation': legal,
             'max_allocation_over_pool': overshoot}
    median = float(np.median(paths['window_surplus'][:, -1]))
    chosen = min(range(len(seeds)), key=lambda i: (abs(float(paths['window_surplus'][i, -1])-median), i))
    selected = branch_episode(paths, details, chosen, arrays, group, origin, seeds)
    zero = selected if chosen == 0 else branch_episode(paths, details, 0, arrays, group, origin, seeds)
    observed = observed_rounds(arrays, group)
    summaries = {name: {'mean': value.mean(axis=0).tolist(), 'p10': np.quantile(value, .1, axis=0).tolist(),
                        'p50': np.quantile(value, .5, axis=0).tolist(), 'p90': np.quantile(value, .9, axis=0).tolist()}
                 for name, value in paths.items() if name in ('pool_after', 'participation', 'window_surplus')}
    bands = [{'round_id': origin+t, 'pool_lower': summaries['pool_after']['p10'][t], 'pool_median': summaries['pool_after']['p50'][t],
              'pool_upper': summaries['pool_after']['p90'][t], 'participation_lower': summaries['participation']['p10'][t],
              'participation_median': summaries['participation']['p50'][t], 'participation_upper': summaries['participation']['p90'][t]} for t in range(horizon)]
    endpoints = {str(h): np.stack((paths['pool_after'][:, h-1]/200, paths['window_surplus'][:, h-1]/(200*h)), axis=-1).tolist() for h in HORIZONS if h <= horizon}
    observed_endpoints = {str(h): [float(arrays['next_pool'][group['index'], origin+h-1])/200,
                                 math.fsum(arrays['surplus'][group['index'], :, origin:origin+h].flat)/(200*h)] for h in HORIZONS if h <= horizon}
    body = {'rounds': observed[:origin]+selected['rounds'], 'observed_rounds': observed, 'simulation_audit': audit,
            'selected_branch': selected, 'branch_zero': zero, 'bands': bands, 'summary_bands': summaries,
            'endpoints': endpoints, 'observed_endpoints': observed_endpoints,
            'prefix_posterior_nodes': details['prefix_posterior_nodes'], 'prefix_posterior_weights': details['prefix_posterior_weights'],
            'parameters': details['parameters'],
            'branch_selection': 'nearest median total retained resources over full window; branch index breaks ties; illustrative only',
            'provenance': {'source': 'Task 05 frozen conditional model from observed human Experiment 1 prefix',
                           'boundary': 'recorded', 'online_parameter_adaptation': False,
                           'posterior': 'resident choices strictly before origin; frozen population parameters',
                           'latent_effect': 'one independent prefix-posterior draw per resident and branch, fixed throughout branch',
                           'branch_predictions': 'condition on this branch sampled resident effect; ensemble bands marginalize latent uncertainty',
                           'trace': 'observed or generated fractions, updated only after simultaneous decisions; peer trace is not an elicited belief'}}
    # Lossless branch representation: seed table + hashes regenerate every field;
    # retain outcomes, actions and latent values directly for independent audits.
    retained = {name: paths[name] for name in ('pool_before', 'pool_after', 'window_surplus', 'participation', 'actual_rounds',
                                               'offers', 'contributions', 'surplus', 'padded', 'sampled_u')}
    return body, retained


def cell_id(record, group, origin, bank):
    return f"task05-{record['family']}-{record['seed']}-g{group['index']}-k{origin}-{bank}-recorded"


def recipes(arrays, manifest, registry, table):
    index = {(r['group_index'], r['origin'], r['family'], r['training_seed'], r['bank']): r['rollout_seeds'] for r in table['rows']}
    for record in registry:
        for group in baseline_groups(manifest):
            for origin in ORIGINS:
                banks = [('main', 20)]
                if origin == 5 and eligible(arrays, group, origin):
                    banks.append(('second', 10))
                for bank, horizon in banks:
                    yield record, group, origin, bank, horizon, index[(group['index'], origin, record['family'], record['seed'], bank)]


def archive_identity(registry, table):
    return {'configuration_sha256': sha256(CONFIG), 'split_sha256': sha256(ROOT / 'results/task03/split.json'),
            'source_sha256': json.loads((ROOT / 'results/task03/split.json').read_text())['source_sha256'],
            'seed_table_hash': table['table_hash'], 'checkpoint_registry': registry,
            'code_sha256': {name: sha256(ROOT / name) for name in ('evopolis/conditional_generate.py', 'evopolis/conditional_models.py',
                'evopolis/forecast_generate.py', 'evopolis/world.py', 'evopolis/behavior_models.py', 'evopolis/behavior_data.py')}}


def generate():
    from .behavior_train import runtime_setup
    runtime_setup()
    arrays, manifest = load()
    table = json.loads((RESULTS / 'forecast_seeds.json').read_text())
    if table['table_hash'] != content_hash(table['rows']):
        raise ValueError('Frozen forecast seeds changed')
    registry = checkpoint_registry()
    completed = json.loads((RESULTS / 'training_complete.json').read_text())
    if completed['fits'] != 12 or completed['epochs_per_fit'] != 480:
        raise ValueError('All twelve fits must complete before forecasts')
    # The root numerical audit must certify integration before opening test scores.
    validate_integration(registry)
    identity = archive_identity(registry, table)
    frozen_path = RESULTS / 'forecast_frozen.json'
    if frozen_path.exists() and json.loads(frozen_path.read_text())['identity'] != identity:
        raise ValueError('Forecast contract changed after freezing')
    if not frozen_path.exists():
        write_json(frozen_path, {'frozen_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'identity': identity,
                                 'branches': 89856, 'principal_branches': 73728, 'second_bank_branches': 16128,
                                 'primary_bank': 'main', 'display_branch': 'branch0'})
    partial = ARCHIVE.with_suffix('.sqlite3.partial')
    target = ARCHIVE if ARCHIVE.exists() else partial
    started, cpu = time.monotonic(), time.process_time()
    database = sqlite3.connect(target)
    generated = skipped = 0
    model_key = None
    all_recipes = list(recipes(arrays, manifest, registry, table))
    if len(all_recipes) != 1404:
        raise ValueError('Forecast branch budget differs')
    try:
        database.execute('CREATE TABLE IF NOT EXISTS cells (id TEXT PRIMARY KEY, metadata TEXT NOT NULL, body BLOB NOT NULL, trajectories BLOB NOT NULL)')
        database.execute('CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
        existing = database.execute("SELECT value FROM metadata WHERE key='identity'").fetchone()
        if existing and json.loads(existing[0]) != identity:
            raise ValueError('Resume rejected: forecast contract changed')
        database.execute("INSERT OR IGNORE INTO metadata VALUES ('identity',?)", (packed(identity),))
        database.commit()
        existing_ids = {r[0] for r in database.execute('SELECT id FROM cells')}
        for number, (record, group, origin, bank, horizon, seeds) in enumerate(all_recipes, 1):
            identifier = cell_id(record, group, origin, bank)
            if identifier in existing_ids:
                skipped += 1
                continue
            key = record['family'], record['seed']
            if key != model_key:
                model, _ = load_checkpoint(record)
                model_key = key
            body, paths = generate_cell(model, arrays, group, origin, seeds, horizon=horizon, nodes=record['integration_nodes'])
            body['provenance']['source_sha256'] = manifest['source_sha256']
            branch = body['selected_branch']
            meta = {**record, 'id': identifier, 'cohort': 'forecast', 'human_id': hashlib.sha256(packed(group['key']).encode()).hexdigest(),
                    'group_index': group['index'], 'key': group['key'], 'condition': group['condition'], 'launch_id': group['launch_id'],
                    'episode_id': group['episode_id'], 'mechanism': group['mechanism'], 'origin': origin, 'horizon': horizon,
                    'bank': bank, 'boundary': 'recorded', 'convention': 'recorded', 'selected_branch_index': branch['branch_index'],
                    'rollout_seed': branch['rollout_seed'], 'source_sha256': manifest['source_sha256'], 'branches': 64,
                    'round_count': origin+horizon, 'actual_rounds': branch['actual_rounds'], 'parameters': body['parameters'],
                    'source_label': 'Conditional-response forecast from observed human history',
                    'provenance_label': 'Observed prefix / frozen conditional model continuation', 'simulation_audit': body['simulation_audit']}
            with database:
                database.execute('INSERT INTO cells VALUES (?,?,?,?)', (identifier, packed(meta), zlib.compress(packed(body).encode(), 6), pack_paths(paths)))
            generated += 1
            if generated == 24:
                measured = {'cells': generated, 'branches': generated*64, 'wall_seconds': time.monotonic()-started,
                            'archive_bytes_so_far': target.stat().st_size,
                            'bytes_per_cell': target.stat().st_size/number,
                            'projected_archive_bytes': target.stat().st_size/number*1404,
                            'projection_limit': 'First 24 completed cells; later horizons, families and compressibility differ',
                            'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                            'selected_chunk_branches': 64, 'scientific_settings_changed': False}
                write_json(RESULTS / 'forecast_benchmark.json', measured)
                print(packed(measured), flush=True)
            if number % 24 == 0 or number == len(all_recipes):
                print(f"Conditional forecast {number}/1404 cells ({number*64:,} branches): {record['family']} seed {record['seed']}; {time.monotonic()-started:.1f}s", flush=True)
        if database.execute('SELECT COUNT(*) FROM cells').fetchone()[0] != 1404:
            raise ValueError('Missing forecast cells')
        totals = {'executed_rounds': 0, 'padded_rounds': 0, 'max_abs_transition_residual': 0.,
                  'max_legal_support_violation': 0., 'max_allocation_over_pool': 0.}
        for encoded, in database.execute('SELECT metadata FROM cells'):
            audit = json.loads(encoded)['simulation_audit']
            for name in totals:
                totals[name] = totals[name]+audit[name] if name.endswith('_rounds') else max(totals[name], audit[name])
        if totals['executed_rounds'] + totals['padded_rounds'] != 1635840:
            raise ValueError('Forecast round budget differs')
        default_group = next(g for g in baseline_groups(manifest) if g['mechanism'] == 'Mixed')
        default_record = next(r for r in registry if r['family'] == 'H1' and r['seed'] == 17)
        provenance = {'identity': identity, 'total_branches': 89856, 'principal_branches': 73728, 'second_bank_branches': 16128,
                      'weights_frozen': True, 'uncertainty_band': '10th to 90th branch percentiles; includes prefix latent uncertainty; not parameter or human-sample uncertainty',
                      'selected_branch': 'nearest median final-window surplus, stable branch index tie break; branch zero also saved'}
        with database:
            database.execute("INSERT OR REPLACE INTO metadata VALUES ('default_id',?)", (packed(cell_id(default_record, default_group, 5, 'main')),))
            database.execute("INSERT OR REPLACE INTO metadata VALUES ('provenance',?)", (packed(provenance),))
        database.execute('VACUUM')
    finally:
        database.close()
    if target == partial:
        partial.replace(ARCHIVE)
    receipt = {'cells': 1404, 'branches': 89856, 'principal_branches': 73728, 'second_bank_branches': 16128,
               'generated_cells_this_invocation': generated, 'resumed_cells': skipped,
               'wall_seconds': time.monotonic()-started, 'cpu_seconds': time.process_time()-cpu,
               'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, 'archive_sha256': sha256(ARCHIVE),
               'archive_bytes': ARCHIVE.stat().st_size, 'simulation_audit': totals, 'identity': identity}
    write_json(RESULTS / 'forecast_generation.json', receipt)
    return {k: v for k, v in receipt.items() if k != 'identity'}


def iter_cells(include_paths=True, path=ARCHIVE):
    with closing(sqlite3.connect(Path(path).resolve().as_uri()+'?mode=ro', uri=True)) as database:
        columns = 'metadata,body,trajectories' if include_paths else 'metadata,body'
        for row in database.execute(f'SELECT {columns} FROM cells ORDER BY id'):
            yield json.loads(row[0]), json.loads(zlib.decompress(row[1])), unpack_paths(row[2]) if include_paths else None


def get_cell(identifier, include_paths=False, path=ARCHIVE):
    from .forecast_generate import get_cell as read
    return read(identifier, include_paths, path)


def verify():
    from .behavior_train import runtime_setup
    runtime_setup()
    arrays, manifest = load()
    table = json.loads((RESULTS / 'forecast_seeds.json').read_text())
    registry = checkpoint_registry()
    with closing(sqlite3.connect(ARCHIVE.resolve().as_uri()+'?mode=ro', uri=True)) as database:
        stored = {k: json.loads(v) for k, v in database.execute('SELECT key,value FROM metadata')}
    if stored['identity'] != archive_identity(registry, table):
        raise ValueError('Forecast source identity changed')
    meta, body, paths = get_cell(stored['default_id'], True)
    record, group, origin, bank, horizon, seeds = next(r for r in recipes(arrays, manifest, registry, table) if cell_id(r[0], r[1], r[2], r[3]) == meta['id'])
    model, _ = load_checkpoint(record)
    rebuilt, regenerated = generate_cell(model, arrays, group, origin, seeds, horizon=horizon, nodes=record['integration_nodes'])
    rebuilt['provenance']['source_sha256'] = manifest['source_sha256']
    if rebuilt != body:
        raise ValueError('Fresh generated body differs from saved archive')
    for name in paths:
        np.testing.assert_array_equal(paths[name], regenerated[name])
    checked = branch_count = 0
    for metadata, saved, trajectories in iter_cells():
        horizon = metadata['horizon']
        if trajectories['pool_after'].shape != (64, horizon):
            raise ValueError('Branch count or horizon differs')
        if np.any(trajectories['contributions'] < 0) or np.any(trajectories['contributions'] > np.floor(trajectories['offers'])):
            raise ValueError('Illegal archived action')
        for i in range(64):
            for t in range(horizon):
                result = step(trajectories['pool_before'][i, t], trajectories['offers'][i, t], trajectories['contributions'][i, t], integer_contributions=True)
                if result.next_pool != trajectories['pool_after'][i, t] or not np.array_equal(result.surplus, trajectories['surplus'][i, t]):
                    raise ValueError('Archived branch fails accounting')
            zero = np.flatnonzero(trajectories['pool_after'][i] == 0)
            if len(zero):
                first = int(zero[0])
                if np.any(trajectories['pool_after'][i, first:] != 0) or np.any(trajectories['window_surplus'][i, first:] != trajectories['window_surplus'][i, first]):
                    raise ValueError('Absorbing-state padding failed')
        for selected in (saved['branch_zero'], saved['selected_branch']):
            np.testing.assert_array_equal(selected['latent_effects'], trajectories['sampled_u'][selected['branch_index']])
            for row in selected['rounds']:
                for p, prediction in enumerate(row['predictions']):
                    if prediction['sampled_latent_effect'] != selected['latent_effects'][p]:
                        raise ValueError('Resident latent effect changed within branch')
                    if not math.isclose(math.fsum(prediction['pmf']), 1., rel_tol=1e-10, abs_tol=1e-10):
                        raise ValueError('Saved illustrated probability mass is not normalized')
        checked += 1
        branch_count += 64
    receipt = {'cells_checked': checked, 'branches_checked': branch_count, 'fresh_process_regenerated_id': meta['id'],
               'fresh_process_branches_exact': 64, 'archive_sha256': sha256(ARCHIVE),
               'source_key': meta['key'], 'displayed_branch_index': 0, 'displayed_branch_seed': body['branch_zero']['rollout_seed'],
               'latent_persistence_verified': True, 'all_archived_actions_and_resource_transitions_verified': True}
    if checked != 1404:
        raise ValueError('Incorrect archive size')
    write_json(RESULTS / 'forecast_verification.json', receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'generate', 'verify'))
    args = parser.parse_args()
    lock_path = ROOT / 'data/cache/task03/training.lock'
    with lock_path.open('w') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = {'prepare': prepare, 'generate': generate, 'verify': verify}[args.command]()
    print(packed(result), flush=True)


if __name__ == '__main__':
    main()
