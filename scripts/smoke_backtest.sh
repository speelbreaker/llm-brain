#!/bin/bash
# Smoke test: Run a minimal backtest (7 days) with synthetic pricing
# Expected: Runs simulation, prints summary metrics

echo "=== Smoke Test: Backtesting Engine (7-day synthetic) ==="
echo ""

python -c "
from datetime import datetime, timezone, timedelta
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
"
