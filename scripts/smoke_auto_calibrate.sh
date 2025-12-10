#!/usr/bin/env bash
set -e

echo "Running auto IV calibration smoke test..."
python scripts/auto_calibrate_iv.py \
  --underlying BTC \
  --dte-min 3 \
  --dte-max 10 \
  --lookback-days 7 \
  --max-samples 500

echo "Smoke test completed."
