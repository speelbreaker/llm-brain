#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/smoke_all.sh --unit   # run full pytest suite (smoke gate)
  bash scripts/smoke_all.sh --all    # run pytest + repo smoke scripts

Options:
  SMOKE_WEB_API=1   Include scripts/smoke_web_api.sh (requires server running)
  PYTHON_BIN=...    Override python executable (default: .venv/bin/python or python)
EOF
}

mode="${1:---unit}"

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -x ".venv/bin/python" ]]; then
    PYTHON_BIN=".venv/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi

run_pytest() {
  echo "=== Smoke: pytest (full suite) ==="
  "$PYTHON_BIN" -m pytest -q
}

run_bash_script() {
  local script="$1"
  if [[ ! -f "$script" ]]; then
    echo "Missing smoke script: $script" >&2
    exit 1
  fi
  echo "=== Smoke: $script ==="
  bash "$script"
}

case "$mode" in
  --unit)
    run_pytest
    ;;
  --all)
    run_pytest
    run_bash_script "scripts/smoke_live_agent.sh"
    run_bash_script "scripts/smoke_backtest.sh"
    run_bash_script "scripts/smoke_training_export.sh"
    run_bash_script "scripts/smoke_reconciliation.sh"
    run_bash_script "scripts/smoke_llm_dataset.sh"

    if [[ "${SMOKE_WEB_API:-0}" == "1" ]]; then
      run_bash_script "scripts/smoke_web_api.sh"
    else
      echo "=== Smoke: scripts/smoke_web_api.sh (skipped) ==="
      echo "Set SMOKE_WEB_API=1 to include it (requires server running)."
    fi

    echo "=== ALL SMOKE TESTS PASSED ==="
    ;;
  -h|--help)
    usage
    exit 0
    ;;
  *)
    echo "Unknown option: $mode" >&2
    usage >&2
    exit 2
    ;;
esac
