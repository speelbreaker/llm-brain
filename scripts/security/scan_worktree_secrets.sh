#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root_dir"

run_gitleaks() {
  gitleaks "$@"
}

if ! command -v gitleaks >/dev/null 2>&1; then
  echo "gitleaks not found locally; using docker image..." >&2
  run_gitleaks() {
    docker run --rm -v "$root_dir":/repo zricethezav/gitleaks:8.18.4 "$@"
  }
fi

run_gitleaks detect \
  --source "$root_dir" \
  --no-git \
  --redact \
  --config "$root_dir/.gitleaks.toml"
