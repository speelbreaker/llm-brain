#!/usr/bin/env python3
"""
Transform candidate-level training CSVs into chat-style JSONL corpora for LLM fine-tuning.

Produces two flavors:
1. Per-candidate classification: One JSONL record per candidate (SELL_CALL or SKIP)
2. Per-decision ranking: One JSONL record per decision_time (pick best index or NO_TRADE)

Usage:
    python scripts/build_llm_training_from_candidates.py \
        --input data/training_candidates_BTC_20190101_20251206_tp_and_roll_20251206_171134.csv \
        --exit-style tp_and_roll \
        --underlying BTC
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def safe_float(val: str | None, field_name: str, row_num: int) -> float:
    """Parse float, raising on invalid data instead of defaulting to 0."""
    if val is None or val.strip() == "":
        raise ValueError(f"Row {row_num}: Missing required field '{field_name}'")
    try:
        return float(val)
    except ValueError:
        raise ValueError(f"Row {row_num}: Invalid float for '{field_name}': {val}")


def safe_int(val: str | None, field_name: str, row_num: int) -> int:
    """Parse int, raising on invalid data instead of defaulting to 0."""
    if val is None or val.strip() == "":
        raise ValueError(f"Row {row_num}: Missing required field '{field_name}'")
    try:
        return int(float(val))
    except ValueError:
        raise ValueError(f"Row {row_num}: Invalid int for '{field_name}': {val}")


def load_candidate_rows(
    path: Path,
    exit_style_filter: str | None = None,
    underlying_filter: str | None = None,
) -> list[dict[str, Any]]:
    """Load candidate CSV and optionally filter by exit_style and underlying."""
    rows = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row_num, row in enumerate(reader, start=2):
            if exit_style_filter and row.get("exit_style") != exit_style_filter:
                continue
            if underlying_filter and row.get("underlying") != underlying_filter:
                continue
            try:
                row["_original_order"] = len(rows)
                row["spot"] = safe_float(row.get("spot"), "spot", row_num)
                row["strike"] = safe_float(row.get("strike"), "strike", row_num)
                row["dte"] = safe_float(row.get("dte"), "dte", row_num)
                row["delta"] = safe_float(row.get("delta"), "delta", row_num)
                row["score"] = safe_float(row.get("score"), "score", row_num)
                row["iv"] = safe_float(row.get("iv"), "iv", row_num)
                row["ivrv_ratio"] = safe_float(row.get("ivrv_ratio"), "ivrv_ratio", row_num)
                row["trade_executed"] = safe_int(row.get("trade_executed"), "trade_executed", row_num)
                row["chosen"] = safe_int(row.get("chosen"), "chosen", row_num)
                row["reward"] = safe_float(row.get("reward"), "reward", row_num)
                row["pnl_vs_hodl"] = safe_float(row.get("pnl_vs_hodl"), "pnl_vs_hodl", row_num)
                row["max_drawdown_pct"] = safe_float(row.get("max_drawdown_pct"), "max_drawdown_pct", row_num)
                rows.append(row)
            except ValueError as e:
                print(f"Warning: Skipping malformed row: {e}")
                continue
    return rows


def build_per_candidate_jsonl(rows: list[dict[str, Any]], out_path: Path) -> int:
    """
    Build per-candidate classification JSONL.
    One record per CSV row with task: decide ACTION = SELL_CALL or SKIP.
    Returns count of records written.
    """
    system_prompt = (
        "You are an automated BTC covered-call policy. "
        "For each candidate option, decide whether to SELL_CALL or SKIP based on the state. "
        "Do not use hindsight PnL to answer; treat it as teacher labels only."
    )

    records = []
    for row in rows:
        user_content = f"""Decision time: {row['decision_time']}
Underlying: {row['underlying']}
Spot price: {row['spot']:.2f}
Teacher policy (exit_style): {row['exit_style']}
Candidate option: {row['instrument']}
  - Strike: {row['strike']:.0f}
  - DTE (days): {row['dte']:.0f}
  - Delta: {row['delta']:.4f}
  - Score: {row['score']:.4f}
  - IV: {row['iv']:.4f}
  - IV/RV ratio: {row['ivrv_ratio']:.4f}

Risk snapshot (teacher hindsight):
  - Max drawdown on this chain: {row['max_drawdown_pct']:.2f}%

PnL outcome for this candidate if chosen (teacher hindsight):
  - Reward (USD): {row['reward']:.2f}
  - PnL vs HODL (USD): {row['pnl_vs_hodl']:.2f}

Question: Based only on the information available at decision time (ignore the realized outcome), what ACTION should the policy take on this candidate? Reply with exactly one token: SELL_CALL or SKIP."""

        action = row.get("action", "")
        if action not in ("SELL_CALL", "SKIP"):
            action = "SELL_CALL" if row.get("chosen") == 1 else "SKIP"

        record = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": action},
            ]
        }
        records.append(record)

    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return len(records)


def build_per_decision_ranking_jsonl(rows: list[dict[str, Any]], out_path: Path) -> int:
    """
    Build per-decision ranking JSONL.
    One record per (decision_time, underlying, exit_style) group.
    Task: pick the best candidate index (1..N) or NO_TRADE.
    Returns count of records written.
    
    Preserves original CSV row ordering within each group to ensure chosen_idx
    matches the teacher's original candidate presentation order.
    """
    system_prompt = (
        "You are an automated BTC covered-call strategy. "
        "At each decision, you see multiple candidate options and may either pick exactly one to SELL_CALL or decide NO_TRADE. "
        "Choose the single best candidate index, imitating the teacher policy."
    )

    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (row["decision_time"], row["underlying"], row["exit_style"])
        groups[key].append(row)

    records = []
    for (decision_time, underlying, exit_style), candidates in groups.items():
        candidates = sorted(candidates, key=lambda x: x.get("_original_order", 0))

        n = len(candidates)
        trade_executed = any(c["trade_executed"] == 1 for c in candidates)
        chosen_idx = None
        chosen_reward = 0.0
        chosen_pnl = 0.0
        has_sell_call_action = False

        for i, c in enumerate(candidates, start=1):
            if c["chosen"] == 1:
                chosen_idx = i
                chosen_reward = c["reward"]
                chosen_pnl = c["pnl_vs_hodl"]
                has_sell_call_action = True
                break
            if c.get("action") == "SELL_CALL":
                has_sell_call_action = True
                if chosen_idx is None:
                    chosen_idx = i
                    chosen_reward = c["reward"]
                    chosen_pnl = c["pnl_vs_hodl"]

        spot = candidates[0]["spot"] if candidates else 0.0

        candidate_lines = []
        for i, c in enumerate(candidates, start=1):
            line = (
                f"  [{i}] {c['instrument']} — "
                f"Strike: {c['strike']:.0f}, "
                f"DTE: {c['dte']:.0f}, "
                f"Delta: {c['delta']:.4f}, "
                f"Score: {c['score']:.4f}"
            )
            candidate_lines.append(line)

        candidates_text = "\n".join(candidate_lines)

        user_content = f"""Decision time: {decision_time}
Underlying: {underlying}
Spot: {spot:.2f}
Teacher policy (exit_style): {exit_style}
Candidates:
{candidates_text}
"""

        if (trade_executed or has_sell_call_action) and chosen_idx is not None:
            user_content += f"""
Outcome (teacher hindsight for the chosen candidate only):
  - Chosen candidate index: {chosen_idx}
  - Reward (USD): {chosen_reward:.2f}
  - PnL vs HODL (USD): {chosen_pnl:.2f}
"""

        user_content += f"""
Question: Before seeing the outcome, which candidate index should the policy choose to SELL_CALL? Reply with exactly one token: 1, 2, ..., {n}, or NO_TRADE."""

        if chosen_idx is not None:
            answer = str(chosen_idx)
        else:
            answer = "NO_TRADE"

        record = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": answer},
            ]
        }
        records.append(record)

    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return len(records)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transform candidate CSV into chat-style JSONL for LLM fine-tuning"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the candidate CSV file",
    )
    parser.add_argument(
        "--exit-style",
        type=str,
        default=None,
        help="Filter by exit_style (e.g., tp_and_roll, hold_to_expiry)",
    )
    parser.add_argument(
        "--underlying",
        type=str,
        default=None,
        help="Filter by underlying (e.g., BTC, ETH)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: same as input file)",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return

    output_dir = Path(args.output_dir) if args.output_dir else input_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = input_path.stem

    print(f"Loading candidates from: {input_path}")
    rows = load_candidate_rows(
        input_path,
        exit_style_filter=args.exit_style,
        underlying_filter=args.underlying,
    )
    print(f"Loaded {len(rows)} candidate rows")

    if not rows:
        print("No rows after filtering. Exiting.")
        return

    per_candidate_path = output_dir / f"{stem}_per_candidate.jsonl"
    per_decision_path = output_dir / f"{stem}_per_decision_ranking.jsonl"

    print(f"\nBuilding per-candidate JSONL...")
    count1 = build_per_candidate_jsonl(rows, per_candidate_path)
    print(f"  Wrote {count1} records to: {per_candidate_path}")

    print(f"\nBuilding per-decision ranking JSONL...")
    count2 = build_per_decision_ranking_jsonl(rows, per_decision_path)
    print(f"  Wrote {count2} records to: {per_decision_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
