"""Sequential, resumable Task 06 entry points and consolidated resource receipt."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT/'results/task06'
COMMANDS = {
    'prepare': ['-m', 'evopolis.institutional_data', '--prepare'],
    'verify-rollout': ['-m', 'evopolis.institutional_rollout', 'verify'],
    'verify-all-rollout': ['-m', 'evopolis.institutional_rollout', 'verify-all'],
    'verify': ['-m', 'evopolis.institutional_verify', 'all'],
    'verify-tilted': ['-m', 'evopolis.institutional_verify', 'tilted-integration'],
    'verify-protected': ['-m', 'evopolis.institutional_verify', 'protected'],
    'references': ['-m', 'evopolis.institutional_rollout', 'references'],
    'train': ['-m', 'evopolis.institutional_train', 'train'],
    'integration': ['-m', 'evopolis.institutional_train', 'verify-integration'],
    'calibrate': ['-m', 'evopolis.institutional_calibrate', 'calibrate'],
    'sensitivity': ['-m', 'evopolis.institutional_calibrate', 'sensitivity'],
    'generate': ['-m', 'evopolis.institutional_rollout', 'all'],
    'audit': ['-m', 'evopolis.institutional_predict', 'audit'],
    'audit-rollouts': ['-m', 'evopolis.institutional_predict', 'audit-rollouts'],
    'diagnostics': ['-m', 'evopolis.institutional_predict', 'diagnostics'],
    'benchmark': ['-m', 'evopolis.institutional_evaluate', 'benchmark'],
    'open-exp2': ['-m', 'evopolis.institutional_data', '--open-exp2'],
    'transfer': ['-m', 'evopolis.institutional_evaluate', 'transfer'],
    'transfer-nll': ['-m', 'evopolis.institutional_predict', 'transfer'],
    'archived-check': ['-m', 'evopolis.institutional_evaluate', 'archived-check'],
    'freeze': ['-m', 'evopolis.institutional_freeze'],
    'plot': ['-m', 'evopolis.institutional_plot'],
    'report': ['-m', 'evopolis.institutional_report'],
    'test': ['-m', 'unittest', 'discover', '-s', 'tests', '-v'],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=list(COMMANDS))
    parser.add_argument('extra', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'
    if not (RESULTS/'resource_current_before_torch.json').exists():
        raise RuntimeError('Run resource profiling before any numerical phase')
    with (RESULTS/'phase.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with tempfile.TemporaryDirectory(prefix='task06-phase-', dir='/tmp') as folder:
            metrics = Path(folder)/'metrics.txt'
            command = [sys.executable, *COMMANDS[args.phase], *args.extra]
            start = time.monotonic()
            code = subprocess.call(['/usr/bin/time', '-f', '%M\n%U\n%S', '-o', str(metrics), *command], cwd=ROOT)
            # GNU time writes a status message before the three numeric lines on failure.
            measured = metrics.read_text().splitlines()[-3:]
            receipt = {'command': command, 'wall_seconds': time.monotonic()-start,
                       'peak_rss_kib': int(measured[0]), 'user_cpu_seconds': float(measured[1]),
                       'system_cpu_seconds': float(measured[2]), 'exit_code': code,
                       'utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
            path = RESULTS/'runtime.json'
            previous = json.loads(path.read_text()) if path.exists() else {}
            previous.setdefault('invocations', {}).setdefault(args.phase, []).append(receipt)
            temp = path.with_suffix('.partial')
            temp.write_text(json.dumps(previous, indent=2)+'\n')
            temp.replace(path)
            print(json.dumps({'phase': args.phase, **receipt}), flush=True)
            return code


if __name__ == '__main__':
    raise SystemExit(main())
