#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root_dir"

if ! command -v pre-commit >/dev/null 2>&1; then
  echo "pre-commit not found. Install with: pip install pre-commit" >&2
  exit 1
fi

pre-commit install --install-hooks
echo "✅ pre-commit hooks installed. Future commits will be scanned for secrets."
