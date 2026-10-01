"""Assemble Task 06's complete pre-Experiment-2 public forecast manifest.

This command does not open the transfer cohort or push a commit. The operator
must commit and push this exact manifest, then record its verified remote SHA.
"""
import datetime
import json
import subprocess

from .behavior_data import ROOT, content_hash, write_json
from .institutional_evaluate import load_rollouts
from .sources import sha256

RESULTS = ROOT/'results/task06'


def _required_evidence(config, checkpoints):
    """Require complete development and verification evidence before opening."""
    evidence = {}

    def read(name):
        path = RESULTS/name
        value = json.loads(path.read_text())
        evidence[str(path.relative_to(ROOT))] = sha256(path)
        return value

    def require(condition, message):
        if not condition:
            raise RuntimeError('Cannot freeze incomplete evidence: '+message)

    families = config['reference_families'] + config['fa_families'] + ['CL-'+f for f in config['calibration_families']]
    seeds = config['training_seeds']
    rules = config['development_rules']
    training = read('training_complete.json')
    features = read('feature_development_audit.json')
    require(features['groups'] == 128 and features['rounds'] == 5120,
            'development feature clipping audit for all128 training/validation groups')
    calibration = read('calibration.json')
    require(training['fits'] == 9 and training['epochs_per_fit'] == 480, 'nine complete 480-epoch fits')
    require(calibration['complete'] and len(calibration['selected']) == 18, 'all18 calibrated lines')
    read('headroom_decision.json')
    benchmark = read('benchmark.json')
    expected_benchmark = {(family, rule) for family in [*families, 'BC1'] for rule in rules}
    for population in ('all', 'train_validation'):
        require({(r['family'], r['rule']) for r in benchmark['populations'][population]['rows']} == expected_benchmark,
                'full Experiment1 benchmark for '+population)
    diagnostics = read('diagnostics_validation.json')
    expected_diagnostics = {(family, seed, rule) for family in families for seed in seeds for rule in [*rules, 'M1']}
    require(diagnostics['checkpoints'] == 57 and diagnostics['human_groups'] == 32
            and {(r['family'], r['seed'], r['mechanism']) for r in diagnostics['per_seed_rule']} == expected_diagnostics
            and all(r['groups'] == 8 and r['scoreable_groups'] == 8 for r in diagnostics['per_seed_rule']),
            '57 checkpoints on all32 validation groups')
    for family in families:
        for seed in seeds:
            saved = read(f'predictions/diagnostics_validation/{family}_{seed}.json')
            require(saved['contract']['checkpoint'] == checkpoints[f'{family}_{seed}']
                    and len(saved['groups']) == 32, 'checkpoint validation predictions')
            for name, digest in saved['contract']['code_sha256'].items():
                require(sha256(ROOT/name) == digest, 'validation prediction code changed: '+name)
    costs = read('calibration_nll_cost.json')
    require(len(costs['per_seed_rule']) == 72 and len(costs['per_seed_all_rules']) == 18
            and len(costs['per_family_rule']) == 24 and len(costs['per_family_all_rules']) == 6,
            'calibration validation-NLL costs for all lines and rules')
    sensitivity_count = 0
    for family in families:
        for seed in seeds:
            for tau in config['sensitivity']['taus']:
                saved = read(f'sensitivity/{family}_{seed}_{tau:+.1f}.json')
                contract = saved['contract']
                require(contract['record'] == checkpoints[f'{family}_{seed}'] and contract['tau'] == tau
                        and set(saved['cells']) == set(rules)
                        and all(saved['cells'][rule]['simulation']['games'] == 256 for rule in rules),
                        'complete sensitivity game counts and checkpoint identities')
                require(contract['config_sha256'] == sha256(ROOT/'configs/task06.json'), 'sensitivity configuration identity')
                for name, digest in contract['code_sha256'].items():
                    require(sha256(ROOT/name) == digest, 'sensitivity code changed: '+name)
                sensitivity_count += 1
    require(sensitivity_count == 285, 'all285 sensitivity cells')

    numerical = {}
    for name in ('integration_checks.json', 'feature_replay_verification.json',
                 'checkpoint_verification.json', 'fresh_process_verification.json',
                 'tilted_integration_verification.json', 'preservation_verification.json',
                 'archived_gru_agreement.json', 'vectorized_verification.json',
                 'verification_complete.json'):
        numerical[name] = read(name)
        require(numerical[name].get('passed') is True, name)
    require(numerical['integration_checks.json']['complete'] and len(numerical['integration_checks.json']['checks']) == 3,
            '41-versus81 integration for all FA-H0 fits')
    require(numerical['feature_replay_verification.json']['rounds'] == 20480,
            'implemented signal checked on all BC1 Interpolating rows')
    require(numerical['checkpoint_verification.json']['unique_fits'] == 39,
            'all39 unique selected checkpoints')
    require(numerical['fresh_process_verification.json']['complete']
            and numerical['fresh_process_verification.json']['families'] == 19,
            'fresh-process regeneration for all19 families')
    require(numerical['tilted_integration_verification.json']['complete']
            and len(numerical['tilted_integration_verification.json']['checks']) == 6,
            'calibrated persistent-model integration')
    vectorized = numerical['vectorized_verification.json']
    require(vectorized['all_families'] and vectorized['code_sha256'] == sha256(ROOT/'evopolis/institutional_rollout.py')
            and {(r['family'], r['rule']) for r in vectorized['checks']} ==
                {(family, rule) for family in families for rule in config['rules']}
            and all(r['exact'] for r in vectorized['checks']), 'vectorized/reference identity for all families/rules')
    require(numerical['verification_complete.json']['complete'], 'full numerical verification sequence')
    read('recorded_interpolating_replay.json')
    read('external_audit_predictions.json')
    audit = read('external_audit_rollouts.json')
    require(len(audit['cells']) == 6 and all(r['count'] == 64 for r in audit['cells']),
            'reported external-audit sensitivity rerun')
    for name in ('diagnostics_validation_groups.csv', 'external_audit_predictions_groups.csv'):
        path = RESULTS/name
        evidence[str(path.relative_to(ROOT))] = sha256(path)

    runtime = json.loads((RESULTS/'runtime.json').read_text())
    tests = runtime.get('invocations', {}).get('test', [])
    require(bool(tests), 'full existing test suite has not run through the phase runner')
    latest = tests[-1]
    require(latest['exit_code'] == 0 and latest['command'][1:] == ['-m', 'unittest', 'discover', '-s', 'tests', '-v'],
            'latest full-suite invocation did not succeed')
    # Runtime continues after the freeze, so retain the exact successful receipt
    # in an immutable scientific snapshot rather than hashing the mutable log.
    snapshot = {'diagnostic_checkpoints': 57, 'validation_groups': 32,
                'sensitivity_cells': sensitivity_count, 'sensitivity_games_per_rule': 256,
                'latest_successful_full_suite': latest,
                'numerical_receipts': {name: evidence[str((RESULTS/name).relative_to(ROOT))] for name in numerical}}
    write_json(RESULTS/'prefreeze_verification.json', snapshot)
    evidence['results/task06/prefreeze_verification.json'] = sha256(RESULTS/'prefreeze_verification.json')
    return evidence


def build():
    if any((RESULTS/name).exists() for name in ('experiment2_games.json', 'experiment2_opening.json')):
        raise RuntimeError('Cannot create a new pre-opening forecast freeze after Experiment2 access')
    config = json.loads((ROOT/'configs/task06.json').read_text())
    training = json.loads((RESULTS/'training_complete.json').read_text())
    calibration = json.loads((RESULTS/'calibration.json').read_text())
    integration = json.loads((RESULTS/'integration_checks.json').read_text())
    decision = json.loads((RESULTS/'headroom_decision.json').read_text())
    if training['fits'] != 9 or not calibration['complete'] or not integration['passed'] or decision['status'] != 'complete':
        raise RuntimeError('Training, calibration, numerical gate and D1 must all complete before freezing')
    _, _, hashes = load_rollouts()
    checkpoints, cells = {}, []
    for path in sorted((RESULTS/'rollouts').glob('*.json')):
        data = json.loads(path.read_text())
        contract = data['contract']
        record = contract['checkpoint']
        if sha256(ROOT/record['path']) != record['sha256']:
            raise RuntimeError('A forecast checkpoint changed')
        for name, digest in contract['code_sha256'].items():
            if sha256(ROOT/name) != digest:
                raise RuntimeError('A forecast was generated by superseded code')
        checkpoints[f"{record['family']}_{record['seed']}"] = record
        cells.append({'file': str(path.relative_to(ROOT)), 'sha256': sha256(path),
                      'trajectory_file': str(path.with_suffix('.npz').relative_to(ROOT)),
                      'trajectory_sha256': data['trajectory_sha256'],
                      'family': record['family'], 'training_seed': record['seed'],
                      'rule': contract['rule'], 'games': contract['count'],
                      'seeds': contract['seeds']})
    if len(cells) != 228 or len(checkpoints) != 57:
        raise RuntimeError('Expected19 families x3 seeds x4 rules, all512 games')
    required_evidence = _required_evidence(config, checkpoints)
    presentation = {'institutional_plot.py', 'institutional_report.py'}
    analysis = {str(p.relative_to(ROOT)): sha256(p) for p in sorted((ROOT/'evopolis').glob('institutional_*.py'))
                if p.name not in presentation}
    for name in ('evopolis/world.py', 'evopolis/behavior_models.py', 'evopolis/conditional_models.py',
                 'evopolis/behavior_data.py', 'scripts/institutional.sh', 'scripts/run-institutional-phase.py',
                 'configs/task06.json'):
        analysis[name] = sha256(ROOT/name)
    value = {'schema_version': 1, 'frozen_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
             'base_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
             'protocol_sha256': sha256(ROOT/'docs/tasks/06-institutional-validity.md'),
             'config_sha256': sha256(ROOT/'configs/task06.json'),
             'source_sha256': json.loads((ROOT/'results/task03/split.json').read_text())['source_sha256'],
             'split_sha256': sha256(ROOT/'results/task03/split.json'),
             'checkpoints': checkpoints, 'calibrated_tilts': calibration['selected'],
             'seed_table': cells, 'forecast_summary_sha256': hashes,
             'games': sum(c['games'] for c in cells), 'D1': decision,
             'fa_best': training['fa_best'],
             'primary_simulators': ['BC1', training['fa_best'], 'CL-'+training['fa_best']],
             'primary_comparisons': config['transfer'],
             'analysis_code_sha256': analysis,
             'required_evidence_sha256': required_evidence,
             'presentation_code_sha256_at_freeze': {str(p.relative_to(ROOT)): sha256(p)
                 for p in sorted((ROOT/'evopolis').glob('institutional_*.py')) if p.name in presentation},
             'recorded_forecasts_sha256': sha256(RESULTS/'recorded_games.json'),
             'known_source_discrepancies_sha256': sha256(RESULTS/'institutional_source_audit.json'),
             'validation_constraints_sha256': sha256(RESULTS/'d1_human_order_constraints.json'),
             'held_closed': ['Experiment2 behavioral rows until this freeze is pushed', 'Experiment3', 'Experiment4 out of scope']}
    value['content_hash'] = content_hash(value)
    path = RESULTS/'manifest.json'
    if path.exists():
        old = json.loads(path.read_text())
        value['frozen_at_utc'] = old['frozen_at_utc']
        value.pop('content_hash')
        value['content_hash'] = content_hash(value)
        if old != value:
            raise RuntimeError('Refusing to overwrite a changed freeze manifest')
    write_json(path, value)
    print(json.dumps({'cells': len(cells), 'games': value['games'], 'fa_best': value['fa_best'],
                      'D1': decision['outcome'], 'manifest_sha256': sha256(path)}))


if __name__ == '__main__':
    build()
