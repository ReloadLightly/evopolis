#!/usr/bin/env bash
set -euo pipefail
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$repo_root/data/cache/uv}"
uv run --frozen python -m evopolis.viewer "$@"
