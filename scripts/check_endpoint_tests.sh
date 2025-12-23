#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

BASE="${GITHUB_BASE_REF:-}"
HEAD="${GITHUB_SHA:-HEAD}"

if [ -z "$BASE" ]; then
  if git rev-parse --verify origin/main >/dev/null 2>&1; then
    BASE="origin/main"
  else
    BASE="HEAD~1"
  fi
fi

ADD_ENDPOINT_PATTERN='@(app|router)\.(get|post|put|delete)\('
if git diff "$BASE" "$HEAD" --unified=0 | grep -E "^\+" | grep -Eq "$ADD_ENDPOINT_PATTERN"; then
  if git diff --name-only "$BASE" "$HEAD" | grep -q '^tests/'; then
    echo "New endpoint detected and tests/ changes found."
    exit 0
  fi
  echo "New endpoint detected between $BASE and $HEAD but no tests/ files changed."
  echo "Add at least one endpoint-level test under tests/ to satisfy the requirement."
  exit 1
fi
echo "No new endpoint decorators detected; skipping endpoint-test enforcement."
