#!/bin/bash
# Smoke test: Run a single iteration of the live agent in dry-run mode
# Expected: Agent fetches state, makes a decision, logs it (no real trades)

echo "=== Smoke Test: Live Agent (1 iteration, dry-run) ==="
echo ""

# Set dry-run mode and limit to 1 iteration
export DRY_RUN=true
export LOOP_INTERVAL_SEC=1

python -c "
from src.config import settings
from src.deribit_client import DeribitClient
from src.state_builder import build_agent_state
from src.policy_rule_based import decide_action
from src.risk_engine import check_action_allowed

print('Connecting to Deribit testnet...')
client = DeribitClient()

print('Building agent state...')
state = build_agent_state(client)

print(f'  Underlyings: {state.underlyings}')
print(f'  Spot prices: {state.spot}')
print(f'  Candidates: {len(state.candidate_options)}')
print(f'  Equity: \${state.portfolio.equity_usd:.2f}')

print('')
print('Making rule-based decision...')
action = decide_action(state)
print(f'  Action: {action.get(\"action\")}')
print(f'  Reasoning: {action.get(\"reasoning\", \"N/A\")[:100]}...')

print('')
print('Checking risk engine...')
allowed, reasons = check_action_allowed(state, action)
print(f'  Allowed: {allowed}')
if reasons:
    print(f'  Reasons: {reasons[:2]}')

print('')
print('=== SMOKE TEST PASSED ===' if state.candidate_options else '=== SMOKE TEST PASSED (no candidates) ===')
"
