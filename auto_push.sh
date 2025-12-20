#!/usr/bin/env bash
set -euo pipefail

msg="${1:-Auto-commit $(date -u +"%Y-%m-%dT%H:%M:%SZ")}" 

if [[ "${SKIP_SMOKE:-0}" != "1" ]]; then
	bash scripts/smoke_all.sh --unit
fi

git add -A

if git diff --cached --quiet; then
	echo "Nothing to commit."
	exit 0
fi

git commit -m "$msg"
git push
