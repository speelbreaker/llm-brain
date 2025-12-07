#!/usr/bin/env bash
#
# Smoke test for position reconciliation logic.
# Tests reconcile_positions function with mock data (no API calls).
#
set -e

echo "=== Smoke Test: Position Reconciliation ==="
echo

cd "$(dirname "$0")/.."

python3 << 'PYEOF'
import sys
sys.path.insert(0, '.')

from src.reconciliation import reconcile_positions, format_reconciliation_summary

print("Test 1: Detect divergence (halt mode)")
print("-" * 40)

exchange_positions = [
    {"symbol": "BTC-20DEC24-100000-C", "size": 0.1, "direction": "sell"},
    {"symbol": "BTC-20DEC24-110000-C", "size": 0.2, "direction": "sell"},
]

local_positions = [
    {"symbol": "BTC-20DEC24-100000-C", "quantity": 0.1},
    {"symbol": "BTC-27DEC24-105000-C", "quantity": 0.15},
]

new_local, stats = reconcile_positions(exchange_positions, local_positions, action="halt")

print(format_reconciliation_summary(stats))
print()

assert stats["divergent"] == True, "Expected divergent=True"
assert "BTC-20DEC24-110000-C" in stats["missing_in_local"], "Expected BTC-20DEC24-110000-C missing in local"
assert "BTC-27DEC24-105000-C" in stats["missing_in_exchange"], "Expected BTC-27DEC24-105000-C missing on exchange"
assert len(new_local) == 2, "halt mode should not modify local positions"

print("PASS: Divergence detected correctly in halt mode")
print()

print("Test 2: Auto-heal mode")
print("-" * 40)

healed, stats2 = reconcile_positions(exchange_positions, local_positions, action="auto_heal")

print(f"Healed positions: {len(healed)}")
for p in healed:
    print(f"  - {p['symbol']}: qty={p['quantity']}")
print()

assert len(healed) == 2, f"Expected 2 healed positions, got {len(healed)}"
healed_symbols = {p["symbol"] for p in healed}
assert "BTC-20DEC24-100000-C" in healed_symbols, "Expected BTC-20DEC24-100000-C in healed"
assert "BTC-20DEC24-110000-C" in healed_symbols, "Expected BTC-20DEC24-110000-C in healed"
assert "BTC-27DEC24-105000-C" not in healed_symbols, "BTC-27DEC24-105000-C should NOT be in healed"

print("PASS: Auto-heal rebuilt positions from exchange")
print()

print("Test 3: Positions in sync")
print("-" * 40)

synced_exchange = [
    {"symbol": "ETH-20DEC24-4000-C", "size": 0.5, "direction": "sell"},
]
synced_local = [
    {"symbol": "ETH-20DEC24-4000-C", "quantity": 0.5},
]

_, stats3 = reconcile_positions(synced_exchange, synced_local, action="halt")

print(format_reconciliation_summary(stats3))
print()

assert stats3["divergent"] == False, "Expected divergent=False for synced positions"
assert len(stats3["missing_in_local"]) == 0, "No missing in local"
assert len(stats3["missing_in_exchange"]) == 0, "No missing on exchange"

print("PASS: Synced positions detected correctly")
print()

print("Test 4: Size mismatch detection")
print("-" * 40)

mismatch_exchange = [
    {"symbol": "BTC-20DEC24-100000-C", "size": 0.2, "direction": "sell"},
]
mismatch_local = [
    {"symbol": "BTC-20DEC24-100000-C", "quantity": 0.1},
]

_, stats4 = reconcile_positions(mismatch_exchange, mismatch_local, action="halt")

print(format_reconciliation_summary(stats4))
print()

assert stats4["divergent"] == True, "Expected divergent=True for size mismatch"
assert len(stats4["size_mismatches"]) == 1, "Expected 1 size mismatch"

print("PASS: Size mismatch detected correctly")
print()

print("=" * 40)
print("ALL RECONCILIATION TESTS PASSED")
print("=" * 40)
PYEOF

echo
echo "=== SMOKE TEST PASSED ==="
