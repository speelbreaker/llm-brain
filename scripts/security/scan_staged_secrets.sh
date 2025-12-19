#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root_dir"

if ! command -v gitleaks >/dev/null 2>&1; then
  echo "gitleaks not found. Install with: brew install gitleaks OR go install github.com/gitleaks/gitleaks/v8@latest" >&2
  exit 1
fi

gitleaks protect \
  --staged \
  --redact \
  --config "$root_dir/.gitleaks.toml"
