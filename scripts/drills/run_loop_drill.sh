#!/bin/bash
set -e

# Configuration
BASE_URL="${BASE_URL:-http://127.0.0.1:8080}"
DRILL_ID="drill-$(date +%s)"
DRILL_BRANCH="drill/$DRILL_ID"
DRILL_FILE="src/_drill_lint.py"

# Ensure we are in the repo
REPO_DIR="$(git rev-parse --show-toplevel)"
cd "$REPO_DIR"

echo ">>> Starting Supervisor Loop Drill: $DRILL_ID"
echo ">>> Base URL: $BASE_URL"

# Check dependencies
if ! command -v gh &> /dev/null; then
    echo "Error: 'gh' is not installed."
    exit 1
fi
if ! command -v jq &> /dev/null; then
    echo "Error: 'jq' is not installed."
    exit 1
fi

# Check Health
echo ">>> Checking Health..."
HEALTH=$(curl -s "$BASE_URL/health")
OK=$(echo "$HEALTH" | jq -r .ok)
if [[ "$OK" != "true" ]]; then
    echo "Error: Supervisor is not healthy: $HEALTH"
    exit 1
fi
echo "Health OK"

# Create Drill Branch
echo ">>> Creating drill branch: $DRILL_BRANCH"
git switch -c "$DRILL_BRANCH"

# Add deterministic lint failure
echo "import os, sys" > "$DRILL_FILE" # Unused imports
echo ">>> Created $DRILL_FILE with lint errors"
git add "$DRILL_FILE"
git commit -m "drill: add lint failure"
git push origin "$DRILL_BRANCH"

# Open PR
echo ">>> Opening PR..."
PR_URL=$(gh pr create --title "Drill: Loop Test $DRILL_ID" --body "Automated drill PR" --head "$DRILL_BRANCH" --base "main")
PR_NUMBER=$(echo "$PR_URL" | grep -oE '[0-9]+$')
echo ">>> PR #$PR_NUMBER opened: $PR_URL"

# Wait for Job
echo ">>> Waiting for Supervisor Job..."
JOB_ID=""
for i in {1..30}; do
    JOBS=$(curl -s "$BASE_URL/api/jobs?limit=5")
    # Find job for this PR
    JOB_ID=$(echo "$JOBS" | jq -r --arg PR "$PR_NUMBER" '.jobs[] | select(.pr_number == ($PR|tonumber)) | .job_id' | head -n 1)
    if [[ -n "$JOB_ID" ]]; then
        break
    fi
    sleep 2
done

if [[ -z "$JOB_ID" ]]; then
    echo "Error: Job not found for PR #$PR_NUMBER"
    exit 1
fi
echo ">>> Job found: $JOB_ID"

# Poll Job Status
echo ">>> Polling Job Status..."
FINAL_STATUS=""
while true; do
    JOB=$(curl -s "$BASE_URL/api/jobs/$JOB_ID")
    STATUS=$(echo "$JOB" | jq -r .job.status)
    echo "Job Status: $STATUS"
    
    if [[ "$STATUS" == "checks_passed" || "$STATUS" == "fixed" || "$STATUS" == "needs_human" || "$STATUS" == "error" ]]; then
        FINAL_STATUS="$STATUS"
        break
    fi
    sleep 5
done

# Check Results
echo ">>> Final Status: $FINAL_STATUS"

DIAG=$(curl -s "$BASE_URL/api/diag")
DRY_RUN=$(echo "$DIAG" | jq -r .dry_run)

if [[ "$DRY_RUN" == "true" ]]; then
    echo ">>> Verified DRY RUN mode."
else
    echo ">>> Verified LIVE mode."
fi

# Cleanup
echo ">>> Cleanup..."
gh pr close "$PR_NUMBER" --delete-branch
git switch main
git branch -D "$DRILL_BRANCH"

echo ">>> Drill Complete."
