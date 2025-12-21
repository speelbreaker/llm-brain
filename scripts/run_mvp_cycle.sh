#!/usr/bin/env bash
set -euo pipefail

# MVP Cycle Runner
# Triggers a manual run of the Testnet Covered Call MVP via the Supervisor API.
# This script is intended to be run via cron or manually on the VPS.

API_URL="${SUPERVISOR_API_URL:-http://127.0.0.1:8080}"

echo "Triggering MVP cycle at $API_URL..."

# Trigger run
response=$(curl -s -X POST "${API_URL}/api/mvp/run")

# Check if successful
if echo "$response" | grep -q '"ok":true'; then
  echo "MVP Cycle Triggered Successfully."
  echo "Response: $response"
  exit 0
else
  echo "MVP Cycle Failed."
  echo "Response: $response"
  exit 1
fi
