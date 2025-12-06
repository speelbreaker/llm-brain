#!/bin/bash
# Smoke test: CSV → JSONL converter for LLM training data
# Expected: Creates JSONL files with proper structure (messages array with system/user/assistant)

set -e

echo "=== Smoke Test: LLM Dataset Converter ==="
echo ""

python3 << 'PYEOF'
import os
import sys
import json
import tempfile
from pathlib import Path

# Step 1: Create a minimal smoke test CSV
csv_content = """decision_time,underlying,spot,instrument,strike,dte,delta,score,iv,ivrv_ratio,exit_style,trade_executed,chosen,action,strategy,reward,pnl_vs_hodl,max_drawdown_pct
2024-01-01T00:00:00+00:00,BTC,42000.0,BTC-08JAN24-44000-C,44000.0,7.0,0.25,3.5,0.65,0.85,hold_to_expiry,1,1,SELL_CALL,conservative,150.0,10.5,0.3
2024-01-01T00:00:00+00:00,BTC,42000.0,BTC-08JAN24-46000-C,46000.0,7.0,0.15,2.1,0.62,0.80,hold_to_expiry,0,0,SKIP,,0.0,0.0,0.0
2024-01-02T00:00:00+00:00,BTC,42500.0,BTC-09JAN24-45000-C,45000.0,7.0,0.22,3.2,0.64,0.82,hold_to_expiry,1,1,SELL_CALL,moderate,120.0,8.2,0.2
2024-01-02T00:00:00+00:00,BTC,42500.0,BTC-09JAN24-47000-C,47000.0,7.0,0.12,1.8,0.60,0.78,hold_to_expiry,0,0,SKIP,,0.0,0.0,0.0
2024-01-03T00:00:00+00:00,BTC,43000.0,BTC-10JAN24-46000-C,46000.0,7.0,0.20,2.9,0.63,0.81,hold_to_expiry,0,0,SKIP,,0.0,0.0,0.0
"""

temp_dir = Path(tempfile.gettempdir())
temp_csv = temp_dir / "smoke_llm_test.csv"
temp_per_candidate = temp_dir / "smoke_per_candidate.jsonl"
temp_per_decision = temp_dir / "smoke_per_decision.jsonl"

temp_csv.write_text(csv_content)
print(f"Created temp CSV: {temp_csv}")
print(f"  Rows: 5 candidate examples (2 SELL_CALL, 3 SKIP)")

# Step 2: Run the converter
print("")
print("Running CSV → JSONL converter...")

sys.path.insert(0, str(Path.cwd()))

from scripts.build_llm_training_from_candidates import (
    load_candidate_rows,
    build_per_candidate_jsonl,
    build_per_decision_ranking_jsonl,
)

rows = load_candidate_rows(temp_csv, exit_style_filter="hold_to_expiry", underlying_filter="BTC")
print(f"  Loaded {len(rows)} candidate rows")

# Build JSONL files
per_cand_count = build_per_candidate_jsonl(rows, temp_per_candidate)
per_dec_count = build_per_decision_ranking_jsonl(rows, temp_per_decision)

print(f"  Per-candidate records written: {per_cand_count}")
print(f"  Per-decision records written: {per_dec_count}")

# Step 3: Validate JSONL structure
print("")
print("Validating JSONL structure...")

errors = []

# Load and check per-candidate JSONL
with open(temp_per_candidate, "r") as f:
    per_cand_records = [json.loads(line) for line in f]

for i, record in enumerate(per_cand_records[:3]):
    if "messages" not in record:
        errors.append(f"Per-candidate record {i}: missing 'messages' key")
    else:
        msgs = record["messages"]
        if not isinstance(msgs, list) or len(msgs) < 3:
            errors.append(f"Per-candidate record {i}: 'messages' should have at least 3 entries")
        else:
            roles = [m.get("role") for m in msgs]
            if roles != ["system", "user", "assistant"]:
                errors.append(f"Per-candidate record {i}: unexpected roles {roles}")
            
            assistant_content = msgs[2].get("content", "")
            if assistant_content not in ["SELL_CALL", "SKIP"]:
                errors.append(f"Per-candidate record {i}: assistant content '{assistant_content}' not in [SELL_CALL, SKIP]")

# Load and check per-decision JSONL
with open(temp_per_decision, "r") as f:
    per_dec_records = [json.loads(line) for line in f]

for i, record in enumerate(per_dec_records[:3]):
    if "messages" not in record:
        errors.append(f"Per-decision record {i}: missing 'messages' key")
    else:
        msgs = record["messages"]
        if not isinstance(msgs, list) or len(msgs) < 3:
            errors.append(f"Per-decision record {i}: 'messages' should have at least 3 entries")
        else:
            roles = [m.get("role") for m in msgs]
            if roles != ["system", "user", "assistant"]:
                errors.append(f"Per-decision record {i}: unexpected roles {roles}")

# Verify JSON serialization round-trip
try:
    json.dumps(per_cand_records[0])
    json.dumps(per_dec_records[0])
    print("  JSON serialization: OK")
except Exception as e:
    errors.append(f"JSON serialization failed: {e}")

# Check label consistency with CSV
chosen_count = sum(1 for r in rows if r.get("chosen") == 1)
sell_call_count = sum(1 for rec in per_cand_records if rec["messages"][2]["content"] == "SELL_CALL")
if chosen_count != sell_call_count:
    errors.append(f"Label mismatch: CSV has {chosen_count} chosen=1, JSONL has {sell_call_count} SELL_CALL")
else:
    print(f"  Label consistency: OK ({chosen_count} SELL_CALL, {len(rows) - chosen_count} SKIP)")

# Print sample record
print("")
print("Sample per-candidate record:")
sample = per_cand_records[0]
print(f"  System: {sample['messages'][0]['content'][:60]}...")
print(f"  User prompt length: {len(sample['messages'][1]['content'])} chars")
print(f"  Assistant: {sample['messages'][2]['content']}")

print("")
print("Sample per-decision record:")
sample = per_dec_records[0]
print(f"  System: {sample['messages'][0]['content'][:60]}...")
print(f"  User prompt length: {len(sample['messages'][1]['content'])} chars")
print(f"  Assistant: {sample['messages'][2]['content']}")

# Cleanup
temp_csv.unlink()
temp_per_candidate.unlink()
temp_per_decision.unlink()
print("")
print("(Temp files cleaned up)")

# Final verdict
print("")
if errors:
    print("=== ERRORS FOUND ===")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
else:
    print("=== LLM DATASET SMOKE TEST PASSED ===")

PYEOF
