#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}/.." rev-parse --show-toplevel)"
DOCS_DIR="${REPO_ROOT}/docs"
OUT="${DOCS_DIR}/TEST_SUMMARY_latest.txt"
mkdir -p "${DOCS_DIR}"

LAST_COMMIT_EPOCH="$(git -C "${REPO_ROOT}" log -1 --format=%ct 2>/dev/null || echo 0)"

if [[ -f "${OUT}" ]]; then
  FILE_EPOCH="$(stat -c %Y "${OUT}" 2>/dev/null || echo 0)"
  if (( FILE_EPOCH >= LAST_COMMIT_EPOCH )); then
    echo "Using existing ${OUT} (newer than last commit)."
    exit 0
  fi
fi

NOW="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
HEAD_SHA="$(git -C "${REPO_ROOT}" rev-parse HEAD 2>/dev/null || echo "unknown")"

{
  echo "generated_at_utc: ${NOW}"
  echo "head_sha: ${HEAD_SHA}"
  echo "summary: No recent pytest summary captured. Agents must run pytest and update this file."
} > "${OUT}"

echo "Wrote ${OUT}"
