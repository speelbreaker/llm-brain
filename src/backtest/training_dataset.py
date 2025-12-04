"""
Training dataset generation for ML/RL models.
Exports (state, action, reward) tuples to CSV/JSON formats.
"""
from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from .covered_call_simulator import CoveredCallSimulator, State, PolicyFn, always_trade_policy
from .types import TrainingExample, CallSimulationConfig
from .deribit_data_source import DeribitDataSource


def generate_training_data(
    sim: CoveredCallSimulator,
    policy: Optional[PolicyFn] = None,
) -> List[TrainingExample]:
    """
    Generate a dataset of (state, action, reward) triples.
    Uses simulate_policy internally.
    """
    return sim.generate_training_data(policy=policy)


def export_to_csv(
    examples: List[TrainingExample],
    filepath: str | Path,
) -> None:
    """
    Export training examples to CSV file.
    """
    if not examples:
        return

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    first = examples[0].to_dict()
    fieldnames = list(first.keys())

    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for ex in examples:
            writer.writerow(ex.to_dict())


def export_to_jsonl(
    examples: List[TrainingExample],
    filepath: str | Path,
) -> None:
    """
    Export training examples to JSON Lines file.
    """
    if not examples:
        return

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex.to_dict()) + "\n")


def compute_dataset_stats(examples: List[TrainingExample]) -> Dict[str, Any]:
    """
    Compute summary statistics for a training dataset.
    """
    if not examples:
        return {"count": 0}

    sell_actions = [ex for ex in examples if ex.action == "SELL_CALL"]
    do_nothing_actions = [ex for ex in examples if ex.action == "DO_NOTHING"]

    rewards = [ex.reward for ex in sell_actions]
    avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
    positive_rewards = sum(1 for r in rewards if r > 0)
    win_rate = positive_rewards / len(rewards) if rewards else 0.0

    return {
        "total_examples": len(examples),
        "sell_call_count": len(sell_actions),
        "do_nothing_count": len(do_nothing_actions),
        "avg_reward": avg_reward,
        "max_reward": max(rewards) if rewards else 0.0,
        "min_reward": min(rewards) if rewards else 0.0,
        "win_rate": win_rate,
        "date_range": {
            "start": min(ex.decision_time for ex in examples).isoformat(),
            "end": max(ex.decision_time for ex in examples).isoformat(),
        },
    }


def run_grid_search(
    data_source: DeribitDataSource,
    base_config: CallSimulationConfig,
    delta_grid: List[float],
    dte_grid: List[int],
) -> List[Dict[str, Any]]:
    """
    Run a simple grid search over delta and DTE parameters.
    Returns results with metrics for each parameter combination.
    """
    results: List[Dict[str, Any]] = []

    for target_delta in delta_grid:
        for target_dte in dte_grid:
            config = CallSimulationConfig(
                underlying=base_config.underlying,
                start=base_config.start,
                end=base_config.end,
                timeframe=base_config.timeframe,
                decision_interval_bars=base_config.decision_interval_bars,
                initial_spot_position=base_config.initial_spot_position,
                contract_size=base_config.contract_size,
                fee_rate=base_config.fee_rate,
                target_dte=target_dte,
                dte_tolerance=base_config.dte_tolerance,
                target_delta=target_delta,
                delta_tolerance=base_config.delta_tolerance,
            )

            sim = CoveredCallSimulator(data_source, config)
            result = sim.simulate_policy(always_trade_policy)

            results.append({
                "target_delta": target_delta,
                "target_dte": target_dte,
                "num_trades": result.metrics["num_trades"],
                "final_pnl": result.metrics["final_pnl"],
                "avg_pnl": result.metrics.get("avg_pnl", 0.0),
                "avg_pnl_vs_hodl": result.metrics.get("avg_pnl_vs_hodl", 0.0),
                "max_drawdown_pct": result.metrics["max_drawdown_pct"],
                "win_rate": result.metrics.get("win_rate", 0.0),
            })

    return results
