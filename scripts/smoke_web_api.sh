#!/usr/bin/env bash
set -e

echo "=== Smoke Test: Web API Status Check ==="
echo ""

BASE_URL="http://localhost:5000"

echo "Testing /health endpoint..."
HEALTH=$(curl -s -w "\n%{http_code}" "$BASE_URL/health")
HTTP_CODE=$(echo "$HEALTH" | tail -n1)
BODY=$(echo "$HEALTH" | sed '$d')

if [ "$HTTP_CODE" != "200" ]; then
    echo "  FAILED: /health returned HTTP $HTTP_CODE"
    exit 1
fi
echo "  OK: HTTP 200"
echo "  Response: $BODY"
echo ""

echo "Testing /status endpoint..."
STATUS=$(curl -s -w "\n%{http_code}" "$BASE_URL/status")
HTTP_CODE=$(echo "$STATUS" | tail -n1)
BODY=$(echo "$STATUS" | sed '$d')

if [ "$HTTP_CODE" != "200" ]; then
    echo "  FAILED: /status returned HTTP $HTTP_CODE"
    exit 1
fi
echo "  OK: HTTP 200"
echo "  Response (truncated): $(echo "$BODY" | head -c 200)..."
echo ""

echo "Testing /api/agent/decisions endpoint..."
DECISIONS=$(curl -s -w "\n%{http_code}" "$BASE_URL/api/agent/decisions")
HTTP_CODE=$(echo "$DECISIONS" | tail -n1)
BODY=$(echo "$DECISIONS" | sed '$d')

if [ "$HTTP_CODE" != "200" ]; then
    echo "  FAILED: /api/agent/decisions returned HTTP $HTTP_CODE"
    exit 1
fi
echo "  OK: HTTP 200"
echo "  Response (truncated): $(echo "$BODY" | head -c 200)..."
echo ""

echo "=== SMOKE TEST PASSED ==="
