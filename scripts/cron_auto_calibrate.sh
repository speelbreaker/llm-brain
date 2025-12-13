#!/bin/bash
#
# Daily auto-calibration cron wrapper script
#
# Usage:
#   ./scripts/cron_auto_calibrate.sh
#   AUTO_CALIBRATE_UNDERLYINGS="BTC,ETH" ./scripts/cron_auto_calibrate.sh
#
# Cron example (run daily at 03:10 UTC):
#   10 3 * * * /path/to/repo/scripts/cron_auto_calibrate.sh >> /path/to/repo/logs/cron_auto_calibrate.log 2>&1
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

LOG_DIR="${PROJECT_ROOT}/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/cron_auto_calibrate.log"

echo "============================================================" | tee -a "$LOG_FILE"
echo "Daily Auto-Calibration - $(date -u +"%Y-%m-%d %H:%M:%S UTC")" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"

UNDERLYINGS="${AUTO_CALIBRATE_UNDERLYINGS:-BTC,ETH}"
DTE_MIN="${AUTO_CALIBRATE_DTE_MIN:-3}"
DTE_MAX="${AUTO_CALIBRATE_DTE_MAX:-10}"
LOOKBACK_DAYS="${AUTO_CALIBRATE_LOOKBACK_DAYS:-14}"
MAX_SAMPLES="${AUTO_CALIBRATE_MAX_SAMPLES:-5000}"
DATA_DIR="${AUTO_CALIBRATE_DATA_DIR:-data/live_deribit}"

echo "Configuration:" | tee -a "$LOG_FILE"
echo "  Underlyings: $UNDERLYINGS" | tee -a "$LOG_FILE"
echo "  DTE range: ${DTE_MIN}-${DTE_MAX}" | tee -a "$LOG_FILE"
echo "  Lookback: ${LOOKBACK_DAYS} days" | tee -a "$LOG_FILE"
echo "  Max samples: ${MAX_SAMPLES}" | tee -a "$LOG_FILE"
echo "  Data dir: ${DATA_DIR}" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

python -m scripts.auto_calibrate_daily \
    --underlyings "$UNDERLYINGS" \
    --dte-min "$DTE_MIN" \
    --dte-max "$DTE_MAX" \
    --lookback-days "$LOOKBACK_DAYS" \
    --max-samples "$MAX_SAMPLES" \
    --data-dir "$DATA_DIR" \
    2>&1 | tee -a "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

echo "" | tee -a "$LOG_FILE"
echo "Exit code: $EXIT_CODE" | tee -a "$LOG_FILE"
echo "Finished: $(date -u +"%Y-%m-%d %H:%M:%S UTC")" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

exit $EXIT_CODE
