#!/usr/bin/env bash
set -euo pipefail

# Apply the repository About text and research topics using an existing gh login.
repo_root=$(git rev-parse --show-toplevel)
origin_url=$(git remote get-url origin)
case "$origin_url" in
  https://github.com/ReloadLightly/evopolis|https://github.com/ReloadLightly/evopolis.git|git@github.com:ReloadLightly/evopolis.git) ;;
  *) printf '%s\n' 'Run this script from the ReloadLightly/evopolis checkout.' >&2; exit 1 ;;
esac

if ! command -v gh >/dev/null 2>&1; then
  printf '%s\n' 'GitHub CLI (gh) is required to apply the repository About settings.' >&2
  exit 1
fi

gh repo edit ReloadLightly/evopolis \
  --description "$(cat "$repo_root/docs/github-description.txt")" \
  --add-topic social-simulation \
  --add-topic computational-social-science \
  --add-topic agent-based-modeling \
  --add-topic world-models \
  --add-topic evolutionary-computation \
  --add-topic cooperation
