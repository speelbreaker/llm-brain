#!/usr/bin/env bash
set -e

msg="${1:-Auto-commit from Replit}"
git add .
git commit -m "$msg" || echo "Nothing to commit."
git push
