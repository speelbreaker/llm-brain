#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
DOCS_DIR="${REPO_ROOT}/docs"
OUT="${DOCS_DIR}/RECENT_DIFF.md"
mkdir -p "${DOCS_DIR}"

# Choose a base ref
if git show-ref --verify --quiet refs/remotes/origin/main; then
  BASE="origin/main"
elif git rev-parse --verify -q HEAD~10 >/dev/null; then
  BASE="HEAD~10"
else
  BASE="$(git rev-list --max-parents=0 HEAD | tail -n 1)"
fi

NOW="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
BRANCH="$(git rev-parse --abbrev-ref HEAD)"
HEAD_SHA="$(git rev-parse HEAD)"

{
  echo "# RECENT_DIFF"
  echo ""
  echo "- generated_at_utc: ${NOW}"
  echo "- branch: ${BRANCH}"
  echo "- head: ${HEAD_SHA}"
  echo "- base: ${BASE}"
  echo ""
  echo "## Last 25 commits"
  git log --oneline -n 25 || true
  echo ""
  echo "## Diff stat (${BASE}..HEAD)"
  git diff --stat "${BASE}..HEAD" || true
  echo ""
  echo "## Patch (${BASE}..HEAD)"
  git diff "${BASE}..HEAD" || true
} > "${OUT}"

# Redact obvious secrets if they show up in diffs
python3 - <<'PY'
import re, pathlib
p = pathlib.Path("docs/RECENT_DIFF.md")
txt = p.read_text(encoding="utf-8", errors="ignore")

patterns = [
  re.compile(r'(OPENAI_API_KEY|DERIBIT_SECRET|DERIBIT_CLIENT_SECRET|API_KEY|SECRET|PASSWORD)\s*[:=]\s*.*', re.I),
  re.compile(r'("access_token"\s*:\s*")[^"]+(")', re.I),
  re.compile(r'("refresh_token"\s*:\s*")[^"]+(")', re.I),
]
for pat in patterns:
  txt = pat.sub(r'\1: [REDACTED]', txt)

lines = txt.splitlines()
MAX = 2000
if len(lines) > MAX:
  lines = lines[:MAX] + ["", "## TRUNCATED", f"Original lines: {len(txt.splitlines())}"]
p.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"Wrote {p}")
PY
