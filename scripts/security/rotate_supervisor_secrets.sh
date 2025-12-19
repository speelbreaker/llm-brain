#!/usr/bin/env bash
set -euo pipefail

# Interactive helper to rotate supervisor secrets safely.

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
env_file="${SUPERVISOR_ENV_FILE:-$root_dir/docker/pr-supervisor.env}"
timestamp=$(date +%Y%m%d-%H%M%S)

if [[ -f "$env_file" ]]; then
  cp "$env_file" "${env_file}.${timestamp}.bak"
  echo "Backup created at ${env_file}.${timestamp}.bak"
else
  echo "Env file not found at $env_file; it will be created."
fi

read -r -p "Enter new OPENAI_API_KEY (or leave blank to skip): " openai_key
read -r -p "Enter new GITHUB_TOKEN (or leave blank to skip): " github_token
read -r -p "Enter new TELEGRAM_BOT_TOKEN (or leave blank to skip): " telegram_token
read -r -p "Enter new GEMINI_API_KEY (or leave blank to skip): " gemini_key

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

if [[ ! -s "$tmp" ]]; then
  echo "No values provided; aborting without changes."
  rm -f "$tmp"
  exit 0
fi

mv "$tmp" "$env_file"
echo "Updated $env_file (redacted). Restart supervisor container to apply."
