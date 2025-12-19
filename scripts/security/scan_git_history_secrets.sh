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

mode="${1:-shallow}"

if [[ "$mode" == "deep" ]]; then
  echo "Running deep history scan (full git history)..."
  run_gitleaks detect --redact --config "$root_dir/.gitleaks.toml" --source "$root_dir" --log-opts="--all"
else
  echo "Running lightweight history scan (current tree)..."
  run_gitleaks detect --redact --config "$root_dir/.gitleaks.toml" --source "$root_dir" --no-git
fi
