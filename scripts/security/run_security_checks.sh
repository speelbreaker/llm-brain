#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root_dir"

TOOL_MISSING_EXIT=2

run_scan() {
  local label="$1"
  shift
  set +e
  "$@"
  local status=$?
  set -e
  if [ "$status" -eq 0 ]; then
    echo "OK: $label"
    return 0
  fi
  if [ "$status" -eq 1 ]; then
    echo "ERROR: leaks detected by $label" >&2
    exit 1
  fi
  if [ "$status" -eq "$TOOL_MISSING_EXIT" ]; then
    if [ "${CI:-}" = "true" ] || [ "${GITHUB_ACTIONS:-}" = "true" ]; then
      echo "ERROR: $label skipped (gitleaks unavailable in CI)" >&2
      exit 1
    fi
    echo "WARN: $label skipped (gitleaks unavailable)" >&2
    return 0
  fi

  echo "ERROR: $label failed with exit code $status" >&2
  exit "$status"
}

run_scans() {
  run_scan "scripts/security/scan_worktree_secrets.sh" scripts/security/scan_worktree_secrets.sh
  run_scan "scripts/security/scan_staged_secrets.sh" scripts/security/scan_staged_secrets.sh
  run_scan "scripts/security/scan_git_history_secrets.sh shallow" scripts/security/scan_git_history_secrets.sh shallow
}

container_running() {
  command -v docker >/dev/null 2>&1 \
    && docker ps -q --filter "name=^/pr-supervisor$" --filter "status=running" | grep -q .
}

run_tests() {
  if container_running; then
    echo "Mode: container"
    echo "Running redaction tests inside pr-supervisor container..."
    if ! docker exec -i -w /app pr-supervisor test -f /app/scripts/secret_tripwire.py; then
      echo "Container missing /app/scripts/secret_tripwire.py; copying from repo..."
      docker exec -i -w /app pr-supervisor mkdir -p /app/scripts
      docker cp "$root_dir/scripts/secret_tripwire.py" pr-supervisor:/app/scripts/secret_tripwire.py
    fi
    docker exec -i -w /app -e PYTHONPATH=/app pr-supervisor python3 -m pytest -q \
      tests/supervisor/test_redact.py tests/security/test_secret_tripwire.py
    return
  fi

  echo "Mode: host-minimal"
  echo "pr-supervisor container not running; using local .venv-security..."
  if [ ! -d ".venv-security" ]; then
    python3 -m venv .venv-security
  fi
  # shellcheck disable=SC1091
  . .venv-security/bin/activate
  python -m pip install pytest
  PYTHONPATH="$root_dir" python -m pytest -q \
    tests/security/test_secret_tripwire.py tests/security/test_redact_minimal.py
  PYTHONPATH="$root_dir" python scripts/secret_tripwire.py
}

run_scans
run_tests
