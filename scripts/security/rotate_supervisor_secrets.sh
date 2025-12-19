#!/usr/bin/env bash
set -euo pipefail

# Interactive helper to rotate supervisor secrets safely.

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
env_file="${SUPERVISOR_ENV_FILE:-$root_dir/docker/pr-supervisor.env}"
timestamp=$(date +%Y%m%d-%H%M%S)
restart_flag=0

if [[ "${1:-}" == "--restart" ]]; then
  restart_flag=1
fi

if [[ -f "$env_file" ]]; then
  cp "$env_file" "${env_file}.${timestamp}.bak"
  echo "Backup created at ${env_file}.${timestamp}.bak"
else
  echo "Env file not found at $env_file; it will be created."
fi

read -s -p "Enter new OPENAI_API_KEY (or leave blank to skip): " openai_key; echo
read -s -p "Enter new GITHUB_TOKEN (or leave blank to skip): " github_token; echo
read -s -p "Enter new TELEGRAM_BOT_TOKEN (or leave blank to skip): " telegram_token; echo
read -s -p "Enter new GEMINI_API_KEY (or leave blank to skip): " gemini_key; echo
read -s -p "Enter new GITHUB_WEBHOOK_SECRET (or leave blank to skip): " webhook_secret; echo

tmp="${env_file}.tmp"
touch "$tmp"

update_var() {
  local var="$1"
  local value="$2"
  if [[ -n "$value" ]]; then
    echo "${var}=${value}" >>"$tmp"
  fi
}

update_var "OPENAI_API_KEY" "$openai_key"
update_var "GITHUB_TOKEN" "$github_token"
update_var "TELEGRAM_BOT_TOKEN" "$telegram_token"
update_var "GEMINI_API_KEY" "$gemini_key"
update_var "GITHUB_WEBHOOK_SECRET" "$webhook_secret"

if [[ ! -s "$tmp" ]]; then
  echo "No values provided; aborting without changes."
  rm -f "$tmp"
  exit 0
fi

mv "$tmp" "$env_file"
chmod 600 "$env_file"

echo "Updated $env_file (values redacted). Keys updated:"
for name in OPENAI_API_KEY GITHUB_TOKEN TELEGRAM_BOT_TOKEN GEMINI_API_KEY GITHUB_WEBHOOK_SECRET; do
  if [[ -n "${!name:-}" ]]; then
    echo "- $name"
  fi
done

if [[ "$restart_flag" -eq 1 ]]; then
  if [[ -x "$root_dir/docker/run_pr_supervisor.sh" ]]; then
    echo "Restarting supervisor via docker/run_pr_supervisor.sh ..."
    "$root_dir/docker/run_pr_supervisor.sh"
  else
    echo "run_pr_supervisor.sh not found or not executable; skipping restart." >&2
  fi
else
  echo "Restart not requested. Run docker/run_pr_supervisor.sh to apply."
fi
