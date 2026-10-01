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
  prepare) uv run --frozen python -m evopolis.behavior_data --prepare "$@" ;;
  benchmark|train) uv run --frozen python -m evopolis.behavior_train "$command" "$@" ;;
  evaluate) uv run --frozen python -m evopolis.behavior_evaluate "$@" ;;
  generate|verify|plot) uv run --frozen python -m evopolis.behavior_generate "$command" "$@" ;;
  *) echo 'Usage: bash scripts/learn.sh {prepare|benchmark|train|evaluate|generate|verify|plot}'; exit 2 ;;
esac
