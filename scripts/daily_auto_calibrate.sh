#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

UNDERLYINGS="${AUTO_CALIBRATE_UNDERLYINGS:-BTC,ETH}"

IFS=',' read -ra UNDERLYING_ARRAY <<< "$UNDERLYINGS"

for underlying in "${UNDERLYING_ARRAY[@]}"; do
    underlying="$(echo "$underlying" | xargs)"
    if [[ -n "$underlying" ]]; then
        python scripts/auto_calibrate_iv.py --underlying "$underlying"
    fi
done
