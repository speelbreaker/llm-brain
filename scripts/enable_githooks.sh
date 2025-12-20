#!/usr/bin/env bash
set -euo pipefail

git config core.hooksPath .githooks

echo "Enabled git hooks via core.hooksPath=.githooks"
echo "Pre-push smoke tests are now active."
