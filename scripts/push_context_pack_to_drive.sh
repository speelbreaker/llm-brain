#!/usr/bin/env bash
set -euo pipefail

EXPECTED_REPO_ROOT="${EXPECTED_REPO_ROOT:-/opt/llm-brain/llm-brain}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}/.." rev-parse --show-toplevel)"

if [[ "${REPO_ROOT}" != "${EXPECTED_REPO_ROOT}" ]]; then
  echo "[FATAL] Wrong repo root: ${REPO_ROOT} (expected ${EXPECTED_REPO_ROOT}). Refusing to publish."
  exit 2
fi

if [[ -d "/root/llm-brain-fix" ]]; then
  WARN_MARKER="/tmp/llmbrain_context_pack_warned"
  if [[ ! -f "${WARN_MARKER}" ]]; then
    echo "[WARN] /root/llm-brain-fix exists. Disable any old timers pointing there."
    touch "${WARN_MARKER}"
  fi
fi

cd "${REPO_ROOT}"

if [[ "${CONTEXT_PACK_PUSH_DIRECT:-}" != "1" ]]; then
  make context-pack-push
  exit $?
fi

REMOTE="${RCLONE_REMOTE:-gdrive}"
FOLDER="${DRIVE_FOLDER:-llm-brain_context_pack}"

# Ensure stable "latest" names exist
cp -f docs/REPO_MANIFEST.json docs/REPO_MANIFEST_latest.json
cp -f docs/RECENT_DIFF.md docs/RECENT_DIFF_latest.md

# Timestamped history snapshots
TS="$(date -u +"%Y%m%d_%H%M%S")"
mkdir -p docs/_context_pack_out
cp -f docs/REPO_MANIFEST.json "docs/_context_pack_out/REPO_MANIFEST_${TS}.json"
cp -f docs/RECENT_DIFF.md    "docs/_context_pack_out/RECENT_DIFF_${TS}.md"

# Upload latest
rclone copyto "docs/REPO_MANIFEST_latest.json" "${REMOTE}:${FOLDER}/REPO_MANIFEST_latest.json"
rclone copyto "docs/REPO_MANIFEST_latest.md"   "${REMOTE}:${FOLDER}/REPO_MANIFEST_latest.md"
rclone copyto "docs/RECENT_DIFF_latest.md"     "${REMOTE}:${FOLDER}/RECENT_DIFF_latest.md"
rclone copyto "docs/ROADMAP_BACKLOG_latest.md" "${REMOTE}:${FOLDER}/ROADMAP_BACKLOG_latest.md"
rclone copyto "docs/TEST_SUMMARY_latest.txt"   "${REMOTE}:${FOLDER}/TEST_SUMMARY_latest.txt"

# Upload history (optional but useful)
rclone copy "docs/_context_pack_out" "${REMOTE}:${FOLDER}/history" --include "*.json" --include "*.md"

echo "Uploaded context pack to Drive folder: ${FOLDER}"
