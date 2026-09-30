#!/usr/bin/env bash
# One command from any working directory; caches stay in the Linux checkout.
set -euo pipefail
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$repo_root/data/cache/uv}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$repo_root/data/cache/matplotlib}"
uv run --frozen python -m evopolis.reproduce "$@"
