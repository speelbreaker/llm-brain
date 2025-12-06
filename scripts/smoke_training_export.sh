#!/bin/bash
# Smoke test: Run a tiny backtest and export training data
# Expected: Creates CSV file, prints path and first few lines

echo "=== Smoke Test: Training Data Export ==="
echo ""

python -c "
from datetime import datetime, timezone, timedelta
from src.backtest.types import CallSimulationConfig
from src.backtest.deribit_data_source import DeribitDataSource
from src.backtest.covered_call_simulator import CoveredCallSimulator
from src.backtest.state_builder import build_historical_state
from src.backtest.training_dataset import (
    DecisionStepData,
    build_candidate_level_examples,
    export_candidate_level_csv,
)
import os

# Use a 3-day window for quick test
end = datetime(2024, 10, 1, tzinfo=timezone.utc)
start = end - timedelta(days=3)

print(f'Running mini-backtest: {start.date()} to {end.date()}')

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
    min_score_to_trade=1.0,
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

# Collect decision step data
decision_steps = []

def state_builder(t):
    return build_historical_state(ds, cfg, t)

for t in decision_times:
    state = state_builder(t)
    candidates = state.get('candidate_options', [])
    spot = state.get('spot')
    
    if spot is None:
        continue
    
    cand_dicts = []
    for c in candidates:
        score = sim._score_candidate(sim._extract_candidate_features(state, c))
        cand_dicts.append({
            'instrument': c.instrument_name,
            'strike': c.strike,
            'dte': (c.expiry - t).total_seconds() / 86400,
            'delta': abs(c.delta) if c.delta else 0,
            'score': score,
            'iv': c.iv,
            'ivrv': c.iv / 0.5 if c.iv else 1.0,
        })
    
    decision_steps.append(DecisionStepData(
        decision_time=t,
        underlying='BTC',
        spot=spot,
        candidates=cand_dicts,
        chosen_hold_to_expiry=None,
        chosen_tp_and_roll=None,
        trade_result_hold=None,
        trade_result_tp=None,
    ))

print(f'Decision steps collected: {len(decision_steps)}')

# Build and export
examples = build_candidate_level_examples(decision_steps, 'hold_to_expiry')
print(f'Candidate examples: {len(examples)}')

# Export to temp file
output_path = 'data/smoke_test_training.csv'
export_candidate_level_csv(examples, output_path)

print('')
print(f'=== Output file: {output_path} ===')
print('')

# Show first few lines
with open(output_path, 'r') as f:
    lines = f.readlines()[:5]
    for line in lines:
        print(line.strip()[:100] + '...' if len(line) > 100 else line.strip())

print('')
print('=== SMOKE TEST PASSED ===')

# Cleanup
os.remove(output_path)
print('(Temp file cleaned up)')
"
