"""Task 06 free communities with batched resident PMFs and indexed randomness.

All accounting goes through the existing allocate/WorldState implementation.
The neural and conditional emissions are reused without changing old artifacts.
Only the first four complete trajectories of each production cell are retained.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import resource
import time

import numpy as np
import torch
from scipy.special import ndtri

from .behavior_data import ROOT, write_json
from .behavior_models import BehaviorModel
from .conditional_models import ConditionalModel, base_log_mass, features, initial_history, update_history
from .institutional_models import features_fa, make_model, signal_step
from .sources import sha256
from .world import WorldState, allocate, participant_observation

RESULTS = ROOT / 'results/task06'
RULES = ('Equal', 'Mixed', 'Proportional', 'Interpolating')
FAMILIES = ('T03-constant', 'T03-linear', 'T03-feedforward', 'T03-GRU',
            'T04-feedforward', 'T04-GRU', 'P0', 'P1', 'H0', 'H1',
            'FA-GRU', 'FA-P0', 'FA-H0')


def registry(include_fa=True, include_cl=False):
    rows = []
    for family in FAMILIES[:13 if include_fa else 10]:
        for seed in (17, 29, 43):
            if family.startswith('T0'):
                task, base = family.split('-')
                base = 'recurrent' if base == 'GRU' else base
                folder = ROOT / f'results/task{task[1:]}/weights/{base}_{seed}'
            elif family.startswith('FA-'):
                base = family
                folder = RESULTS / 'weights' / f'{family}_{seed}'
            else:
                base = family
                folder = ROOT / f'results/task05/weights/{base}_{seed}'
            path = folder / 'best.pt'
            metadata = json.loads((folder / 'metadata.json').read_text())
            digest = sha256(path)
            if metadata['best_sha256'] != digest:
                raise ValueError(f'Checkpoint changed: {path}')
            rows.append({'family': family, 'base_family': base, 'seed': seed,
                         'path': str(path.relative_to(ROOT)), 'sha256': digest,
                         'epoch': metadata['selected_epoch'], 'tilt': [0., 0.]})
    if include_cl:
        calibration = json.loads((RESULTS / 'calibration.json').read_text())
        for row in list(rows):
            key = f"{row['family']}_{row['seed']}"
            if key in calibration['selected']:
                rows.append(dict(row, family='CL-' + row['family'],
                                 tilt=calibration['selected'][key]['tilt']))
    return rows


def load_checkpoint(record):
    path = ROOT / record['path']
    if sha256(path) != record['sha256']:
        raise ValueError('Frozen checkpoint hash changed')
    saved = torch.load(path, map_location='cpu', weights_only=True)
    family = record['base_family']
    if family.startswith('FA-'):
        model = make_model(family)
    elif family in ('P0', 'P1', 'H0', 'H1'):
        model = ConditionalModel(family)
    else:
        model = BehaviorModel(family)
    model.load_state_dict(saved['model_state'])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def seeds_for(family, seed, rule, count, namespace=20261102):
    # CL and its base intentionally share common random numbers. Sensitivity
    # and every calibration grid point also reuse this complete seed plan.
    family = family.removeprefix('CL-')
    index = FAMILIES.index(family)
    return [int(np.random.SeedSequence([namespace, index, seed, RULES.index(rule), i])
                .generate_state(1, dtype=np.uint64)[0]) for i in range(count)]


def uniforms_for(seeds):
    actions = np.empty((len(seeds), 40, 4))
    effects = np.empty((len(seeds), 4))
    for i, seed in enumerate(seeds):
        for p in range(4):
            effects[i, p] = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, p, 0]))).random()
            actions[i, :, p] = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, p, 1]))).random(40)
    return actions, effects


def tilt_log_mass(logp, offers, tilt):
    """Legal-support normalization; zero-offer choices remain point masses."""
    q = torch.arange(logp.shape[-1], dtype=torch.float64) / offers.clamp_min(1)[..., None]
    logits = logp + torch.as_tensor(tilt, dtype=torch.float64)[..., None] * q
    return logits - torch.logsumexp(logits, -1, keepdim=True)


@torch.no_grad()
def simulate(model, rule, seeds, *, tilt=(0., 0.), uniforms=None, retain=4):
    """Vectorize PMFs over games x residents; keep material accounting exact.

    A slow reference is obtained by running each game separately with the very
    same indexed uniforms. Recurrent states and persistent effects never mix.
    """
    if rule not in RULES:
        raise ValueError('Only the four executable baselines are allowed')
    count = len(seeds)
    action_u, effect_u = uniforms_for(seeds) if uniforms is None else uniforms
    if action_u.shape != (count, 40, 4) or effect_u.shape != (count, 4):
        raise ValueError('Random table shape differs from game plan')
    if (not np.isfinite(action_u).all() or not np.isfinite(effect_u).all()
            or np.any(action_u < 0) or np.any(action_u >= 1)
            or np.any(effect_u < 0) or np.any(effect_u >= 1)):
        raise ValueError('Indexed uniforms must be finite and in [0,1)')
    conditional = isinstance(model, ConditionalModel)
    fa = getattr(model, 'task06_fa', False)
    states = [WorldState() for _ in seeds]
    history = initial_history((count,)) if conditional else None
    hidden = torch.zeros(1, count*4, 32) if model.family == 'recurrent' else None
    # PCG64 includes zero. Its inverse-normal endpoint is represented by the
    # smallest positive float, a finite documented numerical convention.
    sampled_u = (ndtri(np.maximum(effect_u, np.nextafter(0., 1.))) * float(model.sigma)) if conditional and model.family.startswith('H') else np.zeros((count, 4))
    signal = torch.full((count,), .5, dtype=torch.float64)
    undefined = torch.ones(count, dtype=torch.bool)
    player_total = np.zeros((count, 4))
    pool20 = np.zeros(count)
    depletion = np.full(count, -1, dtype=np.int16)
    alive_rounds = np.zeros(count, dtype=np.int16)
    kept = min(retain, count)
    paths = {'pool_before': np.zeros((kept, 40)), 'pool_after': np.zeros((kept, 40)),
             'offers': np.zeros((kept, 40, 4)), 'actions': np.zeros((kept, 40, 4), dtype=np.int16),
             'surplus': np.zeros((kept, 40, 4)), 'signal': np.full((kept, 40), .5),
             'undefined': np.ones((kept, 40), dtype=bool), 'padded': np.ones((kept, 40), dtype=bool),
             'sampled_u': sampled_u[:kept].copy()}
    clipping = defined_evaluations = 0
    for t in range(40):
        live = [i for i, state in enumerate(states) if not state.done]
        if not live:
            break
        offers = [allocate(states[i].pool, states[i].previous_contributions, mechanism=rule.lower()) for i in live]
        offer = torch.tensor(offers, dtype=torch.float64)
        previous = torch.tensor([states[i].previous_contributions or (0.,)*4 for i in live], dtype=torch.float64)
        s, undef, clipped = signal_step(offer, previous, signal[live], undefined[live])
        signal[live], undefined[live] = s, undef
        clipping += int(clipped.sum())
        defined_evaluations += int((~undef).sum())
        if conditional:
            selected = {k: v[live] for k, v in history.items()}
            pool = torch.tensor([states[i].pool for i in live], dtype=torch.float64)
            z = features_fa(pool, offer, selected, s, undef) if fa else features(pool, offer, selected)
            raw = model.head(z)
            v = model.beta*(selected['peer_trace']-.5) + torch.tensor(sampled_u[live])
        else:
            observations = [[participant_observation(p, states[i].pool, offers[j], states[i].previous_contributions)
                             for p in range(4)] for j, i in enumerate(live)]
            x = torch.tensor(np.asarray(observations)/200, dtype=torch.float32)
            if fa:
                extra = torch.stack((s, undef.double()), -1).float()[:, None, :].expand(-1, 4, -1)
                x = torch.cat((x, extra), -1)
            residents = np.asarray(live)[:, None]*4 + np.arange(4)
            residents = residents.ravel()
            raw, h = model(x.reshape(len(live)*4, 1, -1), None if hidden is None else hidden[:, residents])
            if hidden is not None:
                hidden[:, residents] = h
            raw = raw[:, 0].reshape(len(live), 4, 5)
            v = torch.zeros_like(offer)
        # Dynamic support saves emission work without changing any legal mass.
        logp = base_log_mass(raw, offer, int(offer.max().floor()))
        logp = tilt_log_mass(logp, offer, v + (float(tilt[0]) + float(tilt[1])*s)[:, None])
        probs = logp.exp().numpy()
        if not np.allclose(probs.sum(-1), 1., rtol=1e-12, atol=1e-12):
            raise FloatingPointError('Unnormalized legal PMF')
        cdf = probs.cumsum(-1)
        # The CDF is mathematically one from the resident's legal maximum,
        # including padded illegal bins. Guard the endpoint at that maximum.
        cdf = np.where(np.arange(cdf.shape[-1]) >= np.floor(offers)[..., None], 1., cdf)
        actions = (cdf <= action_u[live, t, :, None]).sum(-1)
        if np.any(actions > np.floor(offers)):
            raise ValueError('Sampled illegal action')
        if conditional:
            updated = update_history(selected, offer, torch.tensor(actions, dtype=torch.float64), model.eta)
            for key in history:
                history[key][live] = updated[key]
        for j, i in enumerate(live):
            before = states[i]
            states[i], outcome = before.advance(offers[j], actions[j], integer_contributions=True)
            player_total[i] += outcome.surplus
            alive_rounds[i] += 1
            if depletion[i] < 0 and outcome.next_pool < 1:
                depletion[i] = t+1
            if t == 19:
                pool20[i] = outcome.next_pool
            if i < kept:
                paths['pool_before'][i, t] = before.pool
                paths['pool_after'][i, t] = outcome.next_pool
                paths['offers'][i, t] = offers[j]
                paths['actions'][i, t] = actions[j]
                paths['surplus'][i, t] = outcome.surplus
                paths['signal'][i, t] = float(s[j])
                paths['undefined'][i, t] = bool(undef[j])
                paths['padded'][i, t] = False
    rows = []
    for i, state in enumerate(states):
        total = math.fsum(player_total[i])
        gini = None if total == 0 else math.fsum(abs(a-b) for a in player_total[i] for b in player_total[i])/(8*total)
        rows.append({'game': i, 'rollout_seed': str(seeds[i]), 'surplus': total/160,
                     'gini': gini, 'survival': int(state.pool > 1), 'pool20': float(pool20[i]),
                     'pool40': state.pool, 'first_depletion_after_round': int(depletion[i]) if depletion[i] > 0 else None,
                     'actual_rounds': int(alive_rounds[i])})
    return rows, paths, {'clipping_count': clipping, 'defined_signal_evaluations': defined_evaluations}


def simulate_reference(model, rule, seeds, *, tilt=(0., 0.)):
    uniforms = uniforms_for(seeds)
    rows, collected = [], []
    for i, seed in enumerate(seeds):
        row, paths, _ = simulate(model, rule, [seed], tilt=tilt,
                                  uniforms=(uniforms[0][i:i+1], uniforms[1][i:i+1]), retain=1)
        rows.append(dict(row[0], game=i))
        collected.append(paths)
    return rows, {key: np.concatenate([p[key] for p in collected]) for key in collected[0]}


def save_cell(record, rule, count=512, namespace=20261102, folder=None, tilt=None):
    folder = Path(folder or RESULTS/'rollouts')
    folder.mkdir(parents=True, exist_ok=True)
    key = f"{record['family']}_{record['seed']}_{rule}"
    path = folder/f'{key}.json'
    seeds = seeds_for(record['family'], record['seed'], rule, count, namespace)
    contract = {'checkpoint': record, 'rule': rule, 'count': count, 'namespace': namespace,
                'seeds': [str(s) for s in seeds], 'tilt': list(tilt or record['tilt']),
                'code_sha256': {name: sha256(ROOT/name) for name in
                    ('evopolis/institutional_rollout.py', 'evopolis/institutional_models.py',
                     'evopolis/conditional_models.py', 'evopolis/behavior_models.py', 'evopolis/world.py')}}
    if path.exists():
        saved = json.loads(path.read_text())
        if saved['contract'] != contract or sha256(folder/f'{key}.npz') != saved['trajectory_sha256']:
            raise ValueError('Cannot resume a cell with changed contract')
        return saved
    model = load_checkpoint(record)
    start = time.monotonic()
    rows, paths, audit = simulate(model, rule, seeds, tilt=contract['tilt'])
    target = folder/f'{key}.npz'
    with target.with_suffix('.partial').open('wb') as stream:
        np.savez_compressed(stream, **paths)
    target.with_suffix('.partial').replace(target)
    saved = {'contract': contract, 'games': rows, 'audit': audit, 'trajectory_sha256': sha256(target),
             'seconds': time.monotonic()-start, 'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    write_json(path, saved)
    return saved


def main():
    from .behavior_train import runtime_setup
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['references', 'all', 'verify', 'verify-all'])
    args = parser.parse_args()
    runtime_setup()
    records = registry(args.command in ('all', 'verify-all'), args.command in ('all', 'verify-all'))
    if args.command in ('verify', 'verify-all'):
        checks = []
        for record in records:
            if record['seed'] != 17:
                continue
            model = load_checkpoint(record)
            for rule in RULES:
                seeds = seeds_for(record['family'], 17, rule, 5)
                rows, paths, _ = simulate(model, rule, seeds, retain=5, tilt=record['tilt'])
                reference, slow = simulate_reference(model, rule, seeds, tilt=record['tilt'])
                assert rows == reference
                for key in ('actions', 'pool_before', 'pool_after', 'offers', 'surplus'):
                    np.testing.assert_array_equal(paths[key], slow[key])
                checks.append({'family': record['family'], 'rule': rule, 'games': 5, 'exact': True})
        write_json(RESULTS/'vectorized_verification.json', {'checks': checks, 'all_families': args.command == 'verify-all',
                   'code_sha256': sha256(Path(__file__)), 'passed': True})
        return
    start = time.monotonic()
    timings = []
    for index, record in enumerate(records):
        for rule in RULES:
            saved = save_cell(record, rule)
            timings.append(saved['seconds'])
            projected = np.mean(timings)*len(records)*4
            write_json(RESULTS/'rollout_progress.json', {'completed_cells': len(timings),
                'total_cells': len(records)*4, 'projected_seconds': projected,
                'seconds_this_invocation': time.monotonic()-start,
                'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
            print(f"{record['family']}/{record['seed']}/{rule}: {saved['seconds']:.2f}s; projected {projected/3600:.2f}h", flush=True)
            if projected > 28800:
                raise RuntimeError('Declared eight-hour phase projection exceeded; saved cells are resumable')


if __name__ == '__main__':
    main()
