#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}/.." rev-parse --show-toplevel)"
DOCS_DIR="${REPO_ROOT}/docs"
OUT="${DOCS_DIR}/TEST_SUMMARY_latest.txt"
mkdir -p "${DOCS_DIR}"

if (( $# == 0 )); then
  CMD=(python3 -m pytest)
else
  CMD=("$@")
fi

TMP_OUT="$(mktemp)"
set +e
("${CMD[@]}" 2>&1 | tee "${TMP_OUT}")
STATUS=${PIPESTATUS[0]}
set -e

NOW="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
HEAD_SHA="$(git -C "${REPO_ROOT}" rev-parse HEAD 2>/dev/null || echo "unknown")"
SUMMARY_LINE="$(grep -E "==.* in .*s" "${TMP_OUT}" | tail -n 1 | sed -E 's/^=+ //; s/ =+$//')"

{
  echo "generated_at_utc: ${NOW}"
  echo "head_sha: ${HEAD_SHA}"
  echo "command: ${CMD[*]}"
  if [[ -n "${SUMMARY_LINE}" ]]; then
    echo "summary: ${SUMMARY_LINE}"
  else
    echo "summary: pytest summary line not found"
  fi
  echo "exit_code: ${STATUS}"
} > "${OUT}"

if (( STATUS != 0 )); then
  echo "" >> "${OUT}"
  echo "failures:" >> "${OUT}"
  awk '
    BEGIN { capture=0; count=0 }
    /^=+ FAILURES =+/ { capture=1; next }
    capture==1 && count < 20 { print; count++ }
  ' "${TMP_OUT}" >> "${OUT}"
fi

rm -f "${TMP_OUT}"
echo "Wrote ${OUT}"
exit "${STATUS}"
