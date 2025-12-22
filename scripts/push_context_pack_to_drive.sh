#!/usr/bin/env bash
set -euo pipefail

# Push context-pack "latest" artifacts to Google Drive via rclone.
#
# Requirements/assumptions:
# - `rclone` is installed and configured with a remote named `gdrive:`.
# - Target folder path is `gdrive:llm-brain_context_pack` by default.
#
# This script intentionally does not require the web server.

repo_root="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "${repo_root}" ]]; then
  echo "ERROR: not a git repo (cannot determine repo root)" >&2
  exit 2
fi
cd "${repo_root}"

if [[ ! -f "pyproject.toml" ]]; then
  echo "ERROR: refusing to run outside repo root (missing pyproject.toml)" >&2
  exit 2
fi

remote="${CONTEXT_PACK_DRIVE_REMOTE:-gdrive:llm-brain_context_pack}"

if ! command -v rclone >/dev/null 2>&1; then
  echo "ERROR: rclone not found in PATH" >&2
  exit 2
fi

# Generate artifacts first (safe to run repeatedly).
make context-pack-push

mkdir -p docs

# Compatibility: some generators output non-"latest" names.
# We create deterministic *_latest copies for Drive.
cp -f docs/REPO_MANIFEST.json docs/REPO_MANIFEST_latest.json
cp -f docs/RECENT_DIFF.md docs/RECENT_DIFF_latest.md

# If a markdown conversion exists externally, do not overwrite; otherwise provide a stub.
if [[ ! -f docs/REPO_MANIFEST_latest.md ]]; then
  printf "# Repo Manifest\n\nSee REPO_MANIFEST_latest.json\n" > docs/REPO_MANIFEST_latest.md
fi

# Upload the stable set.
files_to_upload=(
  "docs/REPO_MANIFEST_latest.json"
  "docs/REPO_MANIFEST_latest.md"
  "docs/RECENT_DIFF_latest.md"
  "docs/ROADMAP_BACKLOG_latest.md"
  "docs/TEST_SUMMARY_latest.txt"
  "docs/OPS_HEALTH_latest.json"
)

# Add any fidelity latest artifacts if they exist.
shopt -s nullglob
for f in docs/FIDELITY_*_latest.*; do
  files_to_upload+=("$f")
done
shopt -u nullglob

for f in "${files_to_upload[@]}"; do
  if [[ -f "$f" ]]; then
    echo "Uploading $f -> ${remote}/$(basename "$f")"
    rclone copyto "$f" "${remote}/$(basename "$f")"
  else
    echo "WARN: missing expected artifact: $f" >&2
  fi
done

# Optional: upload a timestamped snapshot under history/<timestamp>/
if [[ "${CONTEXT_PACK_UPLOAD_HISTORY:-0}" == "1" ]]; then
  ts="$(date -u +%Y-%m-%dT%H%M%SZ)"
  echo "Uploading history snapshot: ${remote}/history/${ts}/"
  for f in "${files_to_upload[@]}"; do
    if [[ -f "$f" ]]; then
      rclone copyto "$f" "${remote}/history/${ts}/$(basename "$f")"
    fi
  done
fi

echo "Done."
