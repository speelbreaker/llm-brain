#!/usr/bin/env bash
set -e

echo "=== Smoke Test: Web API Status Check ==="
echo ""

BASE_URL="http://localhost:5000"
TIMEOUT_SECONDS=10
PORT=5000

# Check if server is running using curl (POSIX-compatible, no nc dependency)
echo "Checking if FastAPI server is running on port $PORT..."
SERVER_CHECK=$(curl -s --max-time 3 -o /dev/null -w "%{http_code}" "$BASE_URL/health" 2>/dev/null || echo "000")

if [ "$SERVER_CHECK" = "000" ]; then
    echo ""
    echo "ERROR: FastAPI server is not running on port $PORT."
    echo ""
    echo "Please start the web server before running this smoke test:"
    echo "  uvicorn src.web_app:app --host 0.0.0.0 --port 5000"
    echo ""
    echo "Or use the workflow: Options Trading Agent"
    echo ""
    exit 1
fi
echo "  Server detected on port $PORT (HTTP $SERVER_CHECK)"
echo ""

echo "Testing /health endpoint..."
HEALTH=$(curl -s -w "\n%{http_code}" --max-time $TIMEOUT_SECONDS "$BASE_URL/health" 2>&1) || {
    echo "  FAILED: Connection timed out or refused"
    exit 1
}
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
STATUS=$(curl -s -w "\n%{http_code}" --max-time $TIMEOUT_SECONDS "$BASE_URL/status" 2>&1) || {
    echo "  FAILED: Connection timed out or refused"
    exit 1
}
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
DECISIONS=$(curl -s -w "\n%{http_code}" --max-time $TIMEOUT_SECONDS "$BASE_URL/api/agent/decisions" 2>&1) || {
    echo "  FAILED: Connection timed out or refused"
    exit 1
}
HTTP_CODE=$(echo "$DECISIONS" | tail -n1)
BODY=$(echo "$DECISIONS" | sed '$d')

if [ "$HTTP_CODE" != "200" ]; then
    echo "  FAILED: /api/agent/decisions returned HTTP $HTTP_CODE"
    exit 1
fi
echo "  OK: HTTP 200"
echo "  Response (truncated): $(echo "$BODY" | head -c 200)..."
echo ""

echo "Testing /api/agent_healthcheck endpoint..."
HEALTHCHECK=$(curl -s -w "\n%{http_code}" --max-time $TIMEOUT_SECONDS -X POST "$BASE_URL/api/agent_healthcheck" 2>&1) || {
    echo "  FAILED: Connection timed out or refused"
    exit 1
}
HTTP_CODE=$(echo "$HEALTHCHECK" | tail -n1)
BODY=$(echo "$HEALTHCHECK" | sed '$d')

if [ "$HTTP_CODE" != "200" ]; then
    echo "  FAILED: /api/agent_healthcheck returned HTTP $HTTP_CODE"
    exit 1
fi
echo "  OK: HTTP 200"
echo "  Response (truncated): $(echo "$BODY" | head -c 300)..."
echo ""

echo "Testing /api/llm_readiness endpoint..."
LLM_READY=$(curl -s -w "\n%{http_code}" --max-time $TIMEOUT_SECONDS "$BASE_URL/api/llm_readiness" 2>&1) || {
    echo "  FAILED: Connection timed out or refused"
    exit 1
}
HTTP_CODE=$(echo "$LLM_READY" | tail -n1)
BODY=$(echo "$LLM_READY" | sed '$d')

if [ "$HTTP_CODE" != "200" ]; then
    echo "  FAILED: /api/llm_readiness returned HTTP $HTTP_CODE"
    exit 1
fi
echo "  OK: HTTP 200"
echo "  Response: $(echo "$BODY" | head -c 200)..."
echo ""

echo "=== SMOKE TEST PASSED ==="
