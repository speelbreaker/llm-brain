#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}/.." rev-parse --show-toplevel)"
DOCS_DIR="${REPO_ROOT}/docs"
mkdir -p "${DOCS_DIR}"

SOURCE=""
if [[ -f "${REPO_ROOT}/ROADMAP_BACKLOG.md" ]]; then
  SOURCE="${REPO_ROOT}/ROADMAP_BACKLOG.md"
else
  shopt -s nullglob
  candidates=("${REPO_ROOT}"/ROADMAP_BACKLOG*.md)
  shopt -u nullglob
  if (( ${#candidates[@]} > 0 )); then
    SOURCE="$(ls -t "${candidates[@]}" | head -n 1)"
  fi
fi

if [[ -z "${SOURCE}" ]]; then
  echo "[WARN] No ROADMAP_BACKLOG*.md found under ${REPO_ROOT}"
  exit 1
fi

cp -f "${SOURCE}" "${DOCS_DIR}/ROADMAP_BACKLOG_latest.md"
echo "Wrote ${DOCS_DIR}/ROADMAP_BACKLOG_latest.md"
