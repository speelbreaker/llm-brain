#!/bin/bash
# Smoke test: Run a minimal backtest (7 days) with synthetic pricing
# Expected: Runs simulation, prints summary metrics
# Handles Deribit API failures gracefully (403/network blocked)

echo "=== Smoke Test: Backtesting Engine (7-day synthetic) ==="
echo ""

# Check Deribit connectivity using curl (no Python dependencies)
echo "Checking Deribit API connectivity..."
DERIBIT_CHECK=$(curl -s --max-time 10 -o /dev/null -w "%{http_code}" "https://www.deribit.com/api/v2/public/get_index_price?index_name=btc_usd" 2>/dev/null)

if [ "$DERIBIT_CHECK" = "403" ]; then
    echo ""
    echo "WARNING: Deribit API returned 403 (blocked/forbidden)"
    echo ""
    echo "Running in OFFLINE mode - skipping external data fetch."
    echo "This smoke test will be marked as SKIPPED (not a failure)."
    echo ""
    echo "=== SMOKE TEST SKIPPED (Deribit unreachable) ==="
    exit 0
fi

if [ "$DERIBIT_CHECK" != "200" ]; then
    echo ""
    echo "WARNING: Deribit API returned HTTP $DERIBIT_CHECK"
    echo ""
    echo "Running in OFFLINE mode - skipping external data fetch."
    echo "This smoke test will be marked as SKIPPED (not a failure)."
    echo ""
    echo "=== SMOKE TEST SKIPPED (Deribit unreachable) ==="
    exit 0
fi

echo "  Deribit API reachable (HTTP 200)"
echo ""

python -c "
import sys
from datetime import datetime, timezone, timedelta

try:
    from src.backtest.types import CallSimulationConfig
    from src.backtest.deribit_data_source import DeribitDataSource
    from src.backtest.covered_call_simulator import CoveredCallSimulator
    from src.backtest.state_builder import build_historical_state

    # Use a recent 7-day window
    end = datetime(2024, 10, 1, tzinfo=timezone.utc)
    start = end - timedelta(days=7)

    print(f'Running backtest: {start.date()} to {end.date()}')
    print('Underlying: BTC, Pricing: synthetic_bs')
    print('')

    cfg = CallSimulationConfig(
        underlying='BTC',
        start=start,
        end=end,
        timeframe='1h',
        decision_interval_bars=24,
        initial_spot_position=1.0,
        contract_size=1.0,
        fee_rate=0.0003,
        pricing_mode='synthetic_bs',
        synthetic_iv_multiplier=0.62,
        min_score_to_trade=2.0,
    )

    ds = DeribitDataSource()
    sim = CoveredCallSimulator(ds, cfg)

    # Generate decision times
    decision_times = []
    current = start
    while current <= end:
        decision_times.append(current)
        current += timedelta(hours=24)

    print(f'Decision points: {len(decision_times)}')

    # Run scoring-based simulation
    def state_builder(t):
        return build_historical_state(ds, cfg, t)

    result = sim.simulate_policy_with_scoring(
        decision_times=decision_times,
        state_builder=state_builder,
        exit_style='hold_to_expiry',
        min_score_to_trade=2.0,
    )

    print('')
    print('=== Results ===')
    print(f'  Trades executed: {result.metrics.get(\"num_trades\", 0)}')
    print(f'  Final PnL: \${result.metrics.get(\"final_pnl\", 0):.2f}')
    print(f'  Win rate: {result.metrics.get(\"win_rate\", 0)*100:.1f}%')
    print(f'  Max drawdown: {result.metrics.get(\"max_drawdown_pct\", 0):.2f}%')
    print('')
    print('=== SMOKE TEST PASSED ===')

except Exception as e:
    error_msg = str(e).lower()
    if '403' in error_msg or 'forbidden' in error_msg or 'blocked' in error_msg:
        print('')
        print(f'WARNING: Deribit API blocked during execution: {e}')
        print('')
        print('=== SMOKE TEST SKIPPED (Deribit blocked) ===')
        sys.exit(0)
    else:
        print(f'ERROR: {e}')
        sys.exit(1)
"
