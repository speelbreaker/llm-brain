#!/bin/bash
# Wrapper for supervisor loop (CRON/Systemd entrypoint)

# Load env if present
if [ -f .env ]; then
  source .env
fi

# Set defaults if not set
default_repo_dir="${SUPERVISOR_APP_REPO_DIR:-${SUPERVISOR_REPO_DIR:-$(pwd)}}"
export SUPERVISOR_APP_REPO_DIR="${SUPERVISOR_APP_REPO_DIR:-$default_repo_dir}"
export SUPERVISOR_REPO_DIR="${SUPERVISOR_REPO_DIR:-$default_repo_dir}"
export SUPERVISOR_VAULT_REPO_DIR="${SUPERVISOR_VAULT_REPO_DIR:-$default_repo_dir}"
export SUPERVISOR_MODE="${SUPERVISOR_MODE:-dispatch_only}"

# Ensure python path
export PYTHONPATH="${SUPERVISOR_APP_REPO_DIR}:${PYTHONPATH}"

# Run
python3 "${SUPERVISOR_REPO_DIR}/scripts/supervisor_loop.py"
