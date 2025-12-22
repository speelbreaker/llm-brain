#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <pytest args>"
  exit 1
fi

# Allow callers to pass a leading "--" to separate wrapper args.
# Without stripping it, pytest will treat subsequent flags (e.g. -q) as paths.
if [[ "${1:-}" == "--" ]]; then
  shift
fi

tmpfile="$(mktemp)"
python3 -m pytest "$@" | tee "$tmpfile"
status="${PIPESTATUS[0]}"

if [[ "$status" -ne 0 ]]; then
  rm -f "$tmpfile"
  exit "$status"
fi

# Prefer the typical `pytest -q` summary line, e.g.:
#   "7 passed, 4 warnings in 2.05s"
summary_line="$(grep -E "^[0-9]+.*(passed|failed|skipped).*(in [0-9]|seconds)" "$tmpfile" | tail -n 1 || true)"

# Fallback for non-quiet output that includes a banner line.
if [[ -z "$summary_line" ]]; then
  summary_line="$(grep -E "^=+ .* in .*s" "$tmpfile" | tail -n 1 || true)"
fi

summary_line="${summary_line:-pytest completed}"
echo "Pytest summary: ${summary_line}"

# Always refresh the latest test summary artifact for context packs.
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$repo_root/docs"
ts="$(date -u +"%Y-%m-%dT%H:%MZ")"
{
  echo "$ts"
  echo "$summary_line"
} >"$repo_root/docs/TEST_SUMMARY_latest.txt"

if [[ -n "${ROADMAP_APPEND_ARGS:-}" ]]; then
  python3 scripts/roadmap_append_changelog.py ${ROADMAP_APPEND_ARGS} --tests "${summary_line}"
fi

rm -f "$tmpfile"
