"""Task 06 frozen, past-only predictions and institutional attenuation.

Calibration acts on each resident-effect-conditional emission before latent
integration. Its posterior consequently belongs to the calibrated simulator,
not to the original uncalibrated predictor. The transfer entry point verifies
the pushed freeze before opening Experiment 2 through the audited cohort loader.
"""

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import resource
import time

import numpy as np
import torch

from .behavior_data import ROOT, load, write_json
from .conditional_models import (
    ConditionalModel, base_log_mass, features, initial_history, update_history,
)
from .institutional_models import features_fa, signal_sequence
from .institutional_rollout import RESULTS, load_checkpoint, registry
from .sources import sha256

RULES = ('Equal', 'Mixed', 'Proportional')
METRICS = ('nll', 'predicted_fraction', 'observed_fraction', 'p_zero', 'observed_zero')


def _tilted(logp, offers, coefficient, denominator='offer'):
    if denominator not in ('offer', 'floor'):
        raise ValueError('Tilt denominator must be offer or floor')
    divisor = offers if denominator == 'offer' else offers.floor()
    fraction = torch.arange(logp.shape[-1], dtype=torch.float64) / divisor.clamp_min(1)[..., None]
    logits = logp + coefficient[..., None] * fraction
    return logits - torch.logsumexp(logits, -1, keepdim=True)


@torch.no_grad()
def prequential(model, arrays, index, tilt=(0., 0.), nodes=41, *, stop=40,
                return_pmfs=True, tilt_denominator='offer'):
    """Return per-group metrics and optionally round-by-resident PMFs.

    ``arrays`` uses the Task 03 layout [group, resident, round], with its nine
    normalized public inputs in ``x``. All group metrics exclude forced zeros.
    The special floor denominator exists only to reproduce the external audit;
    Task 06 calibration and sensitivity use the recorded offer denominator.
    """
    if not 0 < stop <= arrays['offers'].shape[-1]:
        raise ValueError('Invalid prediction horizon')
    offers = torch.as_tensor(arrays['offers'][index, :, :stop].T, dtype=torch.float64)
    actions = torch.as_tensor(arrays['y'][index, :, :stop].T, dtype=torch.int64)
    signal, undefined, clipping = signal_sequence(offers, actions)
    fa = getattr(model, 'task06_fa', False)
    calibration = float(tilt[0]) + float(tilt[1]) * signal
    posteriors = None

    if isinstance(model, ConditionalModel):
        history = initial_history()
        u, prior = model.latent_grid(nodes)
        logw = prior.expand(4, -1).clone()
        posteriors = [logw.exp().numpy().copy()] if return_pmfs else None
        logs = []
        for t in range(stop):
            pool = torch.tensor(arrays['pool'][index, t], dtype=torch.float64)
            z = (features_fa(pool, offers[t], history, signal[t], undefined[t])
                 if fa else features(pool, offers[t], history))
            base = base_log_mass(model.head(z), offers[t])
            # Peer-response and persistent-effect tilts retain c/e regardless
            # of the external audit's optional calibration denominator.
            v = model.beta * (history['peer_trace'][:, None] - .5) + u
            conditional = _tilted(base[:, None, :], offers[t, :, None], v)
            conditional = _tilted(conditional, offers[t, :, None],
                                  calibration[t].expand_as(v), tilt_denominator)
            marginal = torch.logsumexp(conditional + logw[..., None], 1)
            logs.append(marginal)
            chosen = conditional[torch.arange(4), :, actions[t]]
            logw = logw + chosen
            logw -= torch.logsumexp(logw, -1, keepdim=True)
            if return_pmfs:
                posteriors.append(logw.exp().numpy().copy())
            history = update_history(history, offers[t], actions[t], model.eta)
        logp = torch.stack(logs)
    else:
        x = torch.tensor(arrays['x'][index, :, :stop, :9], dtype=torch.float32)
        if fa:
            extra = torch.stack((signal, undefined.double()), -1).float()
            x = torch.cat((x, extra[None].expand(4, -1, -1)), -1)
        raw, _ = model(x)
        base = base_log_mass(raw.transpose(0, 1), offers)
        logp = _tilted(base, offers, calibration[:, None].expand_as(offers), tilt_denominator)

    pmfs = logp.exp()
    if not torch.allclose(pmfs.sum(-1), torch.ones_like(offers), atol=1e-12, rtol=1e-12):
        raise FloatingPointError('Teacher-forced PMF is not normalized')
    support = torch.arange(201, dtype=torch.float64)
    if torch.any(pmfs.masked_select(support > offers.floor()[..., None]) != 0):
        raise FloatingPointError('Illegal teacher-forced PMF mass')
    observed_logp = logp.gather(-1, actions[..., None]).squeeze(-1)
    expected_fraction = (pmfs * support).sum(-1) / offers.clamp_min(1)
    observed_fraction = actions / offers.clamp_min(1)
    mask = offers >= 1
    count = int(mask.sum())
    summary = {
        'nonforced_choices': count,
        'nll': float(-observed_logp[mask].mean()) if count else None,
        'predicted_fraction': float(expected_fraction[mask].mean()) if count else None,
        'observed_fraction': float(observed_fraction[mask].mean()) if count else None,
        'p_zero': float(pmfs[..., 0][mask].mean()) if count else None,
        'observed_zero': float((actions[mask] == 0).double().mean()) if count else None,
        'signal_clipping_count': int(clipping.sum()),
        'signal_defined_rounds': int((~undefined).sum()),
    }
    result = dict(summary, summary=summary, log_prob=observed_logp.numpy(),
                  expected_fraction=expected_fraction.numpy(),
                  institution_signal=signal.numpy(), institution_undefined=undefined.numpy(),
                  institution_clipping=clipping.numpy())
    if return_pmfs:
        result['pmfs'] = pmfs.numpy()
    if isinstance(model, ConditionalModel):
        result.update(history=history, log_weights=logw, latent_nodes=u.numpy())
        if return_pmfs:
            result['posterior_weights'] = np.asarray(posteriors)
    return result


def group_prediction(model, record, arrays, group, *, tilt=None, nodes=41,
                     tilt_denominator='offer'):
    tilt = record['tilt'] if tilt is None else tilt
    values = prequential(model, arrays, group['index'], tilt, nodes,
                         return_pmfs=False, tilt_denominator=tilt_denominator)['summary']
    return dict(values, family=record['family'], seed=record['seed'],
                checkpoint_sha256=record['sha256'], selected_epoch=record['epoch'],
                group_index=group['index'], key=group['key'],
                mechanism=group['mechanism'], split=group.get('split', 'transfer'),
                tilt=list(tilt), tilt_denominator=tilt_denominator, nodes=nodes)


def predict_group(model, arrays, index, tilt=(0., 0.), nodes=41):
    """Compact interface for calibration and tipping-point diagnostics."""
    return prequential(model, arrays, index, tilt, nodes,
                       return_pmfs=False)['summary']


def summarize(rows):
    """Equal groups within rule and equal fitted seeds; retain seed results."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row['family'], row['seed'], row['mechanism'])].append(row)
    per_seed = []
    for (family, seed, rule), selected in sorted(grouped.items()):
        valid = [r for r in selected if r['nonforced_choices']]
        per_seed.append({'family': family, 'seed': seed, 'mechanism': rule,
                         'groups': len(selected), 'scoreable_groups': len(valid),
                         **{metric: float(np.mean([r[metric] for r in valid])) if valid else None
                            for metric in METRICS}})
    by_family = defaultdict(list)
    for row in per_seed:
        by_family[(row['family'], row['mechanism'])].append(row)
    per_rule = [{'family': family, 'mechanism': rule, 'seeds': len(selected),
                 'groups': selected[0]['groups'],
                 **{metric: float(np.mean([r[metric] for r in selected]))
                    if all(r[metric] is not None for r in selected) else None for metric in METRICS}}
                for (family, rule), selected in sorted(by_family.items())]
    attenuation = []
    for family in sorted({r['family'] for r in per_rule}):
        rules = {r['mechanism']: r for r in per_rule if r['family'] == family}
        if not all(r in rules for r in RULES):
            continue
        e, p = rules['Equal'], rules['Proportional']
        denominator = p['observed_fraction'] - e['observed_fraction']
        attenuation.append({'family': family, 'predicted_P_minus_E': p['predicted_fraction']-e['predicted_fraction'],
                            'observed_P_minus_E': denominator,
                            'attenuation_index': (p['predicted_fraction']-e['predicted_fraction'])/denominator
                            if denominator != 0 else None,
                            'nll_equal_rules': float(np.mean([rules[r]['nll'] for r in RULES]))})
    return {'aggregation': 'Nonforced choices within group, groups within rule, fitted seeds equally; rules equally only for overall summaries',
            'per_seed_rule': per_seed, 'per_rule': per_rule, 'attenuation': attenuation}


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError('No predictions to write')
    temporary = path.with_suffix('.partial')
    with temporary.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, separators=(',', ':')) if isinstance(v, (list, dict)) else v
                             for k, v in row.items()})
    temporary.replace(path)


def calibration_nll_cost(rows):
    """Explicit validation cost for every calibrated line, seed and rule."""
    configuration = json.loads((ROOT / 'configs/task06.json').read_text())
    expected = set(configuration['calibration_families'])
    calibrated = {r['family'][3:] for r in rows if r['family'].startswith('CL-')}
    if calibrated != expected:
        raise ValueError('Calibration validation costs require all six calibrated lines')
    if any(r['split'] != 'validation' for r in rows):
        raise ValueError('Calibration NLL costs must use validation groups only')
    summary = summarize(rows)
    values = {(r['family'], r['seed'], r['mechanism']): r for r in summary['per_seed_rule']}
    rules = (*RULES, 'M1')
    per_seed_rule, per_seed_all = [], []
    for family in sorted(calibrated):
        for seed in (17, 29, 43):
            selected = []
            for rule in rules:
                base = values[family, seed, rule]
                tilted = values['CL-'+family, seed, rule]
                if base['scoreable_groups'] != 8 or tilted['scoreable_groups'] != 8:
                    raise ValueError('Expected eight nonempty validation groups per rule')
                row = {'family': family, 'calibrated_family': 'CL-'+family, 'seed': seed,
                       'rule': rule, 'groups': 8, 'base_nll': base['nll'],
                       'calibrated_nll': tilted['nll'], 'cost': tilted['nll']-base['nll']}
                per_seed_rule.append(row)
                selected.append(row)
            per_seed_all.append({'family': family, 'calibrated_family': 'CL-'+family,
                                 'seed': seed, 'groups': 32,
                                 **{metric: float(np.mean([r[metric] for r in selected]))
                                    for metric in ('base_nll', 'calibrated_nll', 'cost')}})
    family_rule, family_all = [], []
    for family in sorted(calibrated):
        for rule in rules:
            selected = [r for r in per_seed_rule if r['family'] == family and r['rule'] == rule]
            family_rule.append({'family': family, 'calibrated_family': 'CL-'+family,
                                'rule': rule, 'groups': 8, 'seeds': 3,
                                **{metric: float(np.mean([r[metric] for r in selected]))
                                   for metric in ('base_nll', 'calibrated_nll', 'cost')}})
        selected = [r for r in per_seed_all if r['family'] == family]
        family_all.append({'family': family, 'calibrated_family': 'CL-'+family,
                           'groups': 32, 'seeds': 3,
                           **{metric: float(np.mean([r[metric] for r in selected]))
                              for metric in ('base_nll', 'calibrated_nll', 'cost')}})
    return {'definition': 'Calibrated minus base validation nonforced NLL; positive cost means worse likelihood',
            'aggregation': 'Nonforced choices within each group; eight groups per rule; four rules equally including recorded M1 offers; fitted seeds equally',
            'selection_boundary': 'Tilts selected on training outcomes only; these validation costs are descriptive',
            'per_seed_rule': per_seed_rule, 'per_seed_all_rules': per_seed_all,
            'per_family_rule': family_rule, 'per_family_all_rules': family_all}


def diagnostics(records=None, *, split='validation', arrays=None, manifest=None,
                output='diagnostics_validation'):
    """Resume complete checkpoint cells from development or an audited cohort."""
    if arrays is None:
        arrays, manifest = load()
    groups = [g for g in manifest['groups'] if g.get('split') == split]
    records = registry(include_fa=True, include_cl=True) if records is None else records
    started = time.monotonic()
    directory = RESULTS / 'predictions' / output
    directory.mkdir(parents=True, exist_ok=True)
    rows, durations = [], []
    code = {name: sha256(ROOT / name) for name in (
        'evopolis/institutional_predict.py', 'evopolis/institutional_models.py',
        'evopolis/conditional_models.py', 'evopolis/behavior_models.py')}
    for record in records:
        path = directory / f"{record['family']}_{record['seed']}.json"
        contract = {'checkpoint': record, 'group_keys': [g['key'] for g in groups],
                    'split': split, 'nodes': 41, 'code_sha256': code,
                    'source_sha256': manifest.get('source_sha256')}
        if path.exists():
            saved = json.loads(path.read_text())
            if saved['contract'] != contract:
                raise ValueError(f'Prediction resume contract changed: {path}')
        else:
            model = load_checkpoint(record)
            start = time.monotonic()
            chosen = [group_prediction(model, record, arrays, g) for g in groups]
            saved = {'contract': contract, 'groups': chosen, 'seconds': time.monotonic()-start,
                     'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
            write_json(path, saved)
            del model
        rows.extend(saved['groups'])
        durations.append(saved['seconds'])
        projected = float(np.mean(durations) * len(records))
        write_json(RESULTS / f'{output}_progress.json', {
            'completed_checkpoints': len(durations), 'total_checkpoints': len(records),
            'projected_seconds': projected, 'seconds_this_invocation': time.monotonic()-started,
            'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
        print(f"Predictions {record['family']}/{record['seed']}: {saved['seconds']:.2f}s", flush=True)
        if projected > 28800:
            raise RuntimeError('Teacher-forced phase projects beyond eight hours; resume cells saved')
    _write_csv(RESULTS / f'{output}_groups.csv', rows)
    summary = summarize(rows)
    summary.update(seconds_this_invocation=time.monotonic()-started,
                   peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                   checkpoints=len(records), human_groups=len(groups))
    write_json(RESULTS / f'{output}.json', summary)
    if split == 'validation' and any(r['family'].startswith('CL-') for r in rows):
        write_json(RESULTS / 'calibration_nll_cost.json', calibration_nll_cost(rows))
    return rows, summary


def _audit_fraction_variants(result, arrays, group, record):
    """Audit-only weighting/denominator diagnostics; no model recalculation."""
    offers = np.asarray(arrays['offers'][group['index']].T, dtype=float)
    actions = np.asarray(arrays['y'][group['index']].T, dtype=float)
    expected = result['expected_fraction'] * np.maximum(offers, 1)
    mask = offers >= 1
    output = []
    for denominator, divisor in (('actual_offer', offers), ('floored_offer', np.floor(offers))):
        first = len(output)
        predicted = expected / np.maximum(divisor, 1)
        observed = actions / np.maximum(divisor, 1)
        methods = {
            'choices_then_groups': lambda values: float(values[mask].mean()),
            'residents_then_groups': lambda values: float(np.mean([
                values[:, p][mask[:, p]].mean() for p in range(4) if mask[:, p].any()])),
            'rounds_then_groups': lambda values: float(np.mean([
                values[t][mask[t]].mean() for t in range(len(mask)) if mask[t].any()])),
        }
        for method, aggregate in methods.items():
            output.append({'aggregation': method, 'predicted_fraction': aggregate(predicted),
                           'observed_fraction': aggregate(observed)})
        output.append({'aggregation': 'ratio_of_sums_then_groups',
                       'predicted_fraction': float(expected[mask].sum()/divisor[mask].sum()),
                       'observed_fraction': float(actions[mask].sum()/divisor[mask].sum())})
        output.append({'aggregation': 'ratio_of_all_sums_then_groups',
                       'predicted_fraction': float(expected.sum()/divisor.sum()),
                       'observed_fraction': float(actions.sum()/divisor.sum())})
        for row in output[first:]:
            row.update(family=record['family'], seed=record['seed'], mechanism=group['mechanism'],
                       group_index=group['index'], denominator=denominator,
                       nonforced_choices=int(mask.sum()),
                       denominator_sum=float(divisor.sum() if row['aggregation'] == 'ratio_of_all_sums_then_groups'
                                             else divisor[mask].sum()))
    return output


def _audit_fraction_summary(rows):
    """Compare fixed audit alternatives, without promoting one to Task06's estimand."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row['family'], row['seed'], row['mechanism'], row['denominator'], row['aggregation'])].append(row)
    result = []
    for (family, seed, rule, denominator, aggregation), selected in sorted(grouped.items()):
        row = {'family': family, 'seed': seed, 'mechanism': rule,
               'denominator': denominator, 'aggregation': aggregation, 'groups': len(selected),
               **{metric: float(np.mean([r[metric] for r in selected]))
                  for metric in ('predicted_fraction', 'observed_fraction')}}
        result.append(row)
        if aggregation in ('choices_then_groups', 'ratio_of_sums_then_groups', 'ratio_of_all_sums_then_groups'):
            weight = 'nonforced_choices' if aggregation == 'choices_then_groups' else 'denominator_sum'
            total = math.fsum(r[weight] for r in selected)
            pooled_name = {'choices_then_groups': 'pooled_choices', 'ratio_of_sums_then_groups': 'pooled_ratio_of_sums',
                           'ratio_of_all_sums_then_groups': 'pooled_ratio_of_all_sums'}[aggregation]
            result.append(dict(row, aggregation=pooled_name,
                               **{metric: math.fsum(r[metric]*r[weight] for r in selected)/total
                                  for metric in ('predicted_fraction', 'observed_fraction')}))
    return result


def audit():
    """Recompute the external diagnostics on only 24 already-opened test groups.

    Audit-only denominator and aggregation variants locate ambiguous reported
    fractions. They do not replace Task 06's declared actual-offer, group-balanced
    choice means. No rollouts are run by this command.
    """
    arrays, manifest = load()
    groups = [g for g in manifest['groups'] if g['split'] == 'test' and g['mechanism'] in RULES]
    if len(groups) != 24 or any(sum(g['mechanism'] == r for g in groups) != 8 for r in RULES):
        raise ValueError('Audit requires the frozen 24 baseline test groups')
    records = [r for r in registry(include_fa=False) if
               (r['family'] == 'T03-GRU' and r['seed'] == 17) or r['family'] == 'T04-GRU']
    rows, fractions = [], []
    started = time.monotonic()

    def collect(model, record, denominator='offer'):
        for group in groups:
            result = prequential(model, arrays, group['index'], record['tilt'],
                                 return_pmfs=False, tilt_denominator=denominator)
            rows.append(dict(result['summary'], family=record['family'], seed=record['seed'],
                             checkpoint_sha256=record['sha256'], selected_epoch=record['epoch'],
                             group_index=group['index'], key=group['key'], mechanism=group['mechanism'],
                             split=group['split'], tilt=list(record['tilt']),
                             tilt_denominator=denominator, nodes=41))
            fractions.extend(_audit_fraction_variants(result, arrays, group, record))

    for record in records:
        model = load_checkpoint(record)
        collect(model, record)
        if record['family'] == 'T03-GRU':
            tilted = dict(record, family='Audit-T03-GRU-floor-tilt-0.6', tilt=[.6, 0.])
            collect(model, tilted, 'floor')
        del model
    summary = summarize(rows)
    pooled = []
    keys = sorted({(r['family'], r['seed'], r['mechanism']) for r in rows})
    for family, seed, rule in keys:
        chosen = [r for r in rows if (r['family'], r['seed'], r['mechanism']) == (family, seed, rule)]
        n = sum(r['nonforced_choices'] for r in chosen)
        pooled.append({'family': family, 'seed': seed, 'mechanism': rule, 'nonforced_choices': n,
                       **{metric: math.fsum(r[metric]*r['nonforced_choices'] for r in chosen)/n
                          for metric in METRICS}})
    original = next(r for r in summary['attenuation'] if r['family'] == 'T03-GRU')
    tilted = next(r for r in summary['attenuation'] if r['family'] == 'Audit-T03-GRU-floor-tilt-0.6')
    continuation = next(r for r in summary['attenuation'] if r['family'] == 'T04-GRU')
    variant_summary = _audit_fraction_summary(fractions)
    quoted = {'predicted_fraction': {'Equal': .435, 'Mixed': .600, 'Proportional': .710},
              'observed_fraction': {'Equal': .404, 'Mixed': .629, 'Proportional': .745}}
    comparisons = []
    variants = sorted({(r['denominator'], r['aggregation']) for r in variant_summary})
    for denominator, aggregation in variants:
        selected = {r['mechanism']: r for r in variant_summary if r['family'] == 'T03-GRU'
                    and r['seed'] == 17 and (r['denominator'], r['aggregation']) == (denominator, aggregation)}
        differences = {metric: {rule: selected[rule][metric]-quoted[metric][rule] for rule in RULES}
                       for metric in quoted}
        continuation_values = {str(r['seed']): r['predicted_fraction'] for r in variant_summary
                               if r['family'] == 'T04-GRU' and r['mechanism'] == 'Mixed'
                               and (r['denominator'], r['aggregation']) == (denominator, aggregation)}
        tilt_delta = {}
        for rule in RULES:
            tilted_view = next(r for r in variant_summary if r['family'] == 'Audit-T03-GRU-floor-tilt-0.6'
                               and r['mechanism'] == rule and (r['denominator'], r['aggregation']) == (denominator, aggregation))
            tilt_delta[rule] = tilted_view['predicted_fraction']-selected[rule]['predicted_fraction']
        comparisons.append({'denominator': denominator, 'aggregation': aggregation,
                            'actual': {metric: {rule: selected[rule][metric] for rule in RULES} for metric in quoted},
                            'actual_minus_quoted': differences,
                            'maximum_absolute_difference': max(abs(v) for d in differences.values() for v in d.values()),
                            'matches_all_six_at_quoted_three_decimals': all(abs(v) <= .0005 for d in differences.values() for v in d.values()),
                            'continued_mixed_per_seed': continuation_values,
                            'floor_tilt_predicted_fraction_change': tilt_delta})
    matches = [{'denominator': r['denominator'], 'aggregation': r['aggregation']} for r in comparisons
               if r['matches_all_six_at_quoted_three_decimals']]
    interpretation = ('The brief fractions reproduce under the listed alternative estimand(s); Task06 primary diagnostics retain actual-offer choice means within groups.'
                      if matches else 'None of the declared audit alternatives reproduces all six quoted fractions at their stated precision; preserve the discrepancy and Task06 primary estimand.')
    summary.update(
        cohort='24 opened Exp1 baseline test groups; never used for Task06 selection',
        original_audit_tilt='exp(0.6*c/floor(e)); distinct from Task06 c/e calibration',
        choice_pooled_secondary=pooled,
        floor_tilt_nll_cost_equal_rules=tilted['nll_equal_rules']-original['nll_equal_rules'],
        continuation_nll_change_equal_rules=continuation['nll_equal_rules']-original['nll_equal_rules'],
        continuation_comparison='T04 three-seed mean versus specified T03 seed17; individual seeds retained',
        fraction_audit={'quoted': quoted, 'comparisons': comparisons, 'matching_variants': matches,
                        'interpretation': interpretation,
                        'variant_definitions': {
                            'choices_then_groups': 'Average nonforced c/divisor within each group, then equal groups; actual_offer is Task06 primary',
                            'residents_then_groups': 'Average nonforced c/divisor within each resident, then equal residents within group, then groups',
                            'rounds_then_groups': 'Average c/divisor across nonforced residents in each active round, then active rounds within group, then groups',
                            'ratio_of_sums_then_groups': 'Sum returns divided by sum divisor over nonforced choices in each group, then equal groups; allocation-weighted when divisor=e',
                            'ratio_of_all_sums_then_groups': 'Sum returns divided by sum divisor over all160 choices in each group, including forced-choice allocations; then equal groups',
                            'pooled_choices': 'Equal nonforced choices pooled across groups',
                            'pooled_ratio_of_sums': 'Total returns divided by total divisor across every nonforced choice in the rule',
                            'pooled_ratio_of_all_sums': 'Total returns divided by total divisor across all choices in the rule, including forced-choice allocations'},
                        'forced_handling': 'All-sums variants include allocations on forced choices; every other variant excludes forced choices. Task06 primary is unchanged.',
                        'per_seed_rule': variant_summary, 'per_group': fractions},
        survival_64_games_status='Not run by this teacher-forced audit; requires separate explicitly scheduled rollout audit',
        seconds=time.monotonic()-started,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    _write_csv(RESULTS / 'external_audit_predictions_groups.csv', rows)
    write_json(RESULTS / 'external_audit_predictions.json', summary)
    return summary


def audit_rollouts():
    """New 64-game audit draws; external audit's random stream is unavailable.

    A local emission adapter changes only the audit tilt denominator. It is
    installed in a scoped patch because the production simulator deliberately
    exposes only the preregistered actual-offer calibration convention.
    """
    from unittest.mock import patch
    from . import institutional_rollout as rollout

    record = next(r for r in registry(include_fa=False)
                  if r['family'] == 'T03-GRU' and r['seed'] == 17)
    model = load_checkpoint(record)
    rows = []
    started = time.monotonic()

    def audit_floor_mass(logp, offers, coefficient):
        return _tilted(logp, offers, torch.as_tensor(coefficient, dtype=torch.float64), 'floor')

    for rule in RULES:
        seeds = rollout.seeds_for('T03-GRU', 17, rule, 64, 20261102)
        for tau in (0., .6):
            start = time.monotonic()
            if tau:
                with patch.object(rollout, 'tilt_log_mass', audit_floor_mass):
                    games, _, checks = rollout.simulate(model, rule, seeds,
                                                       tilt=(tau, 0.), retain=0)
            else:
                games, _, checks = rollout.simulate(model, rule, seeds,
                                                   tilt=(0., 0.), retain=0)
            row = {'rule': rule, 'tau': tau, 'tilt_denominator': 'floor',
                   'games': games, 'checks': checks, 'count': 64,
                   'seconds': time.monotonic()-start}
            for metric in ('surplus', 'survival', 'pool20', 'pool40'):
                values = np.asarray([g[metric] for g in games], dtype=float)
                row[metric] = float(values.mean())
                row[metric + '_mcse'] = float(values.std(ddof=1)/math.sqrt(len(values)))
            rows.append(row)
            write_json(RESULTS / 'external_audit_rollouts_progress.json', {
                'completed_cells': len(rows), 'total_cells': 6, 'cells': rows,
                'seconds': time.monotonic()-started})
            print(f"Audit 64 draws {rule}/tau={tau}: survival {row['survival']:.4f}", flush=True)
    result = {'checkpoint': record, 'namespace': 20261102, 'count_per_cell': 64,
              'streams': 'First64 Task06 per-rule streams; paired across audit tilts',
              'scope': 'Diagnostic only; no fitting, family selection or calibration',
              'reproduction_limit': 'External audit seed was not supplied; these are new draws with Monte Carlo error, not exact replication of its64-game bank',
              'cells': rows, 'seconds': time.monotonic()-started,
              'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    write_json(RESULTS / 'external_audit_rollouts.json', result)
    return result


def main():
    from .behavior_train import runtime_setup
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('diagnostics', 'transfer', 'audit', 'audit-rollouts'))
    parser.add_argument('--references-only', action='store_true')
    args = parser.parse_args()
    runtime_setup()
    if args.command == 'audit':
        audit()
    elif args.command == 'audit-rollouts':
        audit_rollouts()
    elif args.command == 'transfer':
        from .institutional_data import load_exp2
        cohort = load_exp2()  # remote-pushed freeze gate precedes numeric parsing
        if len(cohort.groups) != 120:
            raise ValueError('Transfer likelihood requires all120 Experiment2 groups')
        manifest = {'source_sha256': cohort.audit['source_sha256'],
                    'groups': [dict(g, split='transfer') for g in cohort.groups]}
        diagnostics(split='transfer', arrays=cohort.arrays, manifest=manifest,
                    output='diagnostics_experiment2')
    else:
        records = registry(include_fa=False) if args.references_only else None
        diagnostics(records)


if __name__ == '__main__':
    main()
