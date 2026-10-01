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
  benchmark-train|train|verify-continuation) uv run --frozen python -m evopolis.forecast_train "$command" "$@" ;;
  prepare|audit|benchmark|generate|verify) uv run --frozen python -m evopolis.forecast_generate "$command" "$@" ;;
  observations|plot) uv run --frozen python -m evopolis.forecast_evaluate "$command" "$@" ;;
  scores) uv run --frozen python -m evopolis.forecast_evaluate forecasts "$@" ;;
  *) echo 'Usage: bash scripts/forecast.sh {benchmark-train|verify-continuation|train|prepare|audit|benchmark|observations|generate|scores|verify|plot}'; exit 2 ;;
esac
