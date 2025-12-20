#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root_dir"

echo "Branch: $(git rev-parse --abbrev-ref HEAD)"
echo "HEAD: $(git rev-parse HEAD)"

python3 scripts/security/env_safety_gate.py
./scripts/security/run_security_checks.sh

if command -v docker >/dev/null 2>&1 \
  && docker ps -q --filter "name=^/pr-supervisor$" --filter "status=running" | grep -q .; then
  echo "Supervisor container detected; running API checks."

  health_code="$(curl -sS -o /tmp/health.txt -w "%{http_code}" http://127.0.0.1:8080/health || true)"
  if [ "$health_code" != "200" ]; then
    echo "ERROR: /health returned $health_code"
    exit 1
  fi

  diag_code="$(curl -sS -o /tmp/diag.txt -w "%{http_code}" http://127.0.0.1:8080/api/diag || true)"
  if [ "$diag_code" != "200" ]; then
    echo "ERROR: /api/diag returned $diag_code"
    exit 1
  fi
  if command -v rg >/dev/null 2>&1; then
    if rg -n -i "sk-|bearer |api_key|token|secret" /tmp/diag.txt >/dev/null 2>&1; then
      echo "ERROR: /api/diag output contains forbidden substrings."
      exit 1
    fi
  elif grep -E -i "sk-|bearer |api_key|token|secret" /tmp/diag.txt >/dev/null 2>&1; then
    echo "ERROR: /api/diag output contains forbidden substrings."
    exit 1
  fi

  debug_code="$(curl -sS -o /tmp/debug.txt -w "%{http_code}" -X POST http://127.0.0.1:8080/debug/simulate_pr_event || true)"
  if [ "$debug_code" != "404" ]; then
    echo "ERROR: /debug/simulate_pr_event returned $debug_code (expected 404)."
    exit 1
  fi
else
  echo "Supervisor container not running; skipping API checks."
fi

echo "Audit complete: OK"
