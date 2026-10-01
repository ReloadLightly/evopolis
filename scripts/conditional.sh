#!/usr/bin/env bash
set -euo pipefail
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$repo_root/data/cache/uv}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
command=${1:-help}
if [[ $# -gt 0 ]]; then shift; fi
case "$command" in
  profile) python3 - <<'PY'
import datetime,json,pathlib,shutil,os
path=pathlib.Path('data/cache/task05/resource-latest.json')
path.parent.mkdir(parents=True,exist_ok=True)
report={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'meminfo':pathlib.Path('/proc/meminfo').read_text(),'disk_free_bytes':shutil.disk_usage('.').free,'cpus':os.cpu_count(),'threads':1,'microbatch_groups':1}
path.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
PY
    ;;
  train) uv run --frozen python -m evopolis.conditional_train "$@" ;;
  integration) uv run --frozen python -m evopolis.conditional_integration "$@" ;;
  prepare|generate|verify) uv run --frozen python -m evopolis.conditional_generate "$command" "$@" ;;
  observations) uv run --frozen python -m evopolis.conditional_evaluate observations "$@" ;;
  scores) uv run --frozen python -m evopolis.conditional_evaluate forecasts "$@" ;;
  diagnostics|plot) uv run --frozen python -m evopolis.conditional_diagnostics "$command" "$@" ;;
  audit) uv run --frozen python scripts/check-task05-scores.py "$@" ;;
  *) echo 'Usage: bash scripts/conditional.sh {profile|train|integration|prepare|observations|generate|scores|verify|diagnostics|plot|audit}'; exit 2 ;;
esac
