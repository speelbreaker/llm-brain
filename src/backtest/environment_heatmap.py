"""
Environment Occupancy Heatmap module.

Computes strategy-agnostic environment occupancy and strategy-conditional
pass statistics for Greg strategies on metric pairs.

This is an offline backtest / analysis tool, not part of the live trading loop.
"""
from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from src.backtest.selector_scan import (
    _generate_synthetic_sensors,
    _check_strategy_passes,
    GREG_STRATEGIES,
    GREG_STRATEGY_CRITERIA,
)


MetricSample = Dict[str, float]

AVAILABLE_METRICS = [
    "vrp_30d",
    "chop_factor_7d",
    "iv_rank_6m",
    "term_structure_spread",
    "skew_25d",
    "adx_14d",
    "rsi_14d",
    "price_vs_ma200",
]

STRATEGY_LABELS = {
    "STRATEGY_A_STRADDLE": "Strategy A: ATM Straddle",
    "STRATEGY_A_STRANGLE": "Strategy B: OTM Strangle",
    "STRATEGY_B_CALENDAR": "Strategy C: Calendar Spread",
    "STRATEGY_C_SHORT_PUT": "Strategy D: Short Put",
    "STRATEGY_D_IRON_BUTTERFLY": "Strategy E: Iron Butterfly",
    "STRATEGY_F_BULL_PUT_SPREAD": "Strategy F: Bull Put Spread",
    "STRATEGY_F_BEAR_CALL_SPREAD": "Strategy G: Bear Call Spread",
}

LABEL_TO_KEY = {v: k for k, v in STRATEGY_LABELS.items()}


@dataclass
class HeatmapCell:
    """A single cell in an environment heatmap."""
    x_low: float
    x_high: float
    y_low: float
    y_high: float
    count: int
    occupancy_frac: float


@dataclass
class StrategyHeatmapCell(HeatmapCell):
    """A heatmap cell with Greg strategy-conditional statistics."""
    total_samples_in_cell: int
    strategy_selected_count: int
    strategy_pass_frac: float
    avg_score: Optional[float] = None


@dataclass
class SweetSpot:
    """A sweet spot region for a strategy."""
    x_low: float
    x_high: float
    y_low: float
    y_high: float
    occupancy_frac: float
    strategy_pass_frac: float
    sweetness: float


@dataclass
class StrategySweetSpotSummary:
    """Summary of sweet spots for a strategy on a metric pair."""
    underlying: str
    strategy: str
    strategy_key: str
    x_metric: str
    y_metric: str
    sweet_spots: List[SweetSpot]


def load_metric_samples_for_underlying(
    underlying: str,
    lookback_days: int = 365,
    decision_interval_days: int = 1,
) -> Iterable[MetricSample]:
    """
    Yield historical metric snapshots for the given underlying.
    
    Uses the same synthetic data that drives the Bots tab / Greg sensors.
    """
    num_steps = int(lookback_days / decision_interval_days)
    seed = hash((underlying, lookback_days)) % (2**31)
    rng = np.random.default_rng(seed)
    
    for step_idx in range(num_steps):
        sensors = _generate_synthetic_sensors(underlying, step_idx, rng)
        yield {k: v for k, v in sensors.items() if v is not None}


def compute_environment_heatmap(
    samples: Iterable[MetricSample],
    x_metric: str,
    y_metric: str,
    x_bins: int = 20,
    y_bins: int = 20,
    x_range: Optional[Tuple[float, float]] = None,
    y_range: Optional[Tuple[float, float]] = None,
) -> List[HeatmapCell]:
    """
    Build a 2D occupancy histogram for (x_metric, y_metric).
    
    Args:
        samples: iterable of metric dicts, all from a single underlying.
        x_metric, y_metric: names like "vrp_30d", "adx_14d", etc.
        x_bins, y_bins: number of buckets per axis.
        x_range, y_range: optional overrides; if None, infer from data.
    
    Returns:
        A flat list of HeatmapCell with count and occupancy_frac.
    """
    samples_list = list(samples)
    
    x_vals = []
    y_vals = []
    for s in samples_list:
        xv = s.get(x_metric)
        yv = s.get(y_metric)
        if xv is not None and yv is not None:
            x_vals.append(xv)
            y_vals.append(yv)
    
    if not x_vals:
        return []
    
    if x_range is None:
        x_min, x_max = min(x_vals), max(x_vals)
        margin = (x_max - x_min) * 0.05 if x_max > x_min else 1.0
        x_range = (x_min - margin, x_max + margin)
    
    if y_range is None:
        y_min, y_max = min(y_vals), max(y_vals)
        margin = (y_max - y_min) * 0.05 if y_max > y_min else 1.0
        y_range = (y_min - margin, y_max + margin)
    
    x_step = (x_range[1] - x_range[0]) / x_bins
    y_step = (y_range[1] - y_range[0]) / y_bins
    
    counts = [[0 for _ in range(x_bins)] for _ in range(y_bins)]
    total = 0
    
    for xv, yv in zip(x_vals, y_vals):
        ix = int((xv - x_range[0]) / x_step) if x_step > 0 else 0
        iy = int((yv - y_range[0]) / y_step) if y_step > 0 else 0
        
        ix = max(0, min(x_bins - 1, ix))
        iy = max(0, min(y_bins - 1, iy))
        
        counts[iy][ix] += 1
        total += 1
    
    cells = []
    for iy in range(y_bins):
        for ix in range(x_bins):
            x_low = x_range[0] + ix * x_step
            x_high = x_low + x_step
            y_low = y_range[0] + iy * y_step
            y_high = y_low + y_step
            count = counts[iy][ix]
            occupancy_frac = count / total if total > 0 else 0.0
            
            cells.append(HeatmapCell(
                x_low=round(x_low, 4),
                x_high=round(x_high, 4),
                y_low=round(y_low, 4),
                y_high=round(y_high, 4),
                count=count,
                occupancy_frac=round(occupancy_frac, 6),
            ))
    
    return cells


def evaluate_greg_strategy_for_sample(
    metrics: MetricSample,
    underlying: str,
) -> Dict[str, Any]:
    """
    Given a single metric snapshot, evaluate all Greg strategies.
    
    Returns a dict with:
        - selected_strategy: the first strategy that passes (or "NO_TRADE")
        - strategy_statuses: dict of {strategy_key: bool} for all strategies
    """
    strategy_statuses = {}
    selected_strategy = "NO_TRADE"
    
    for strategy_key in GREG_STRATEGIES:
        if strategy_key == "NO_TRADE":
            continue
        passes = _check_strategy_passes(metrics, strategy_key, {})
        strategy_statuses[strategy_key] = passes
        if passes and selected_strategy == "NO_TRADE":
            selected_strategy = strategy_key
    
    return {
        "selected_strategy": selected_strategy,
        "strategy_statuses": strategy_statuses,
    }


def compute_strategy_heatmap_for_pair(
    samples: Iterable[MetricSample],
    x_metric: str,
    y_metric: str,
    strategy_key: str,
    underlying: str,
    x_bins: int = 20,
    y_bins: int = 20,
    x_range: Optional[Tuple[float, float]] = None,
    y_range: Optional[Tuple[float, float]] = None,
) -> List[StrategyHeatmapCell]:
    """
    For the given strategy_key, compute:
    - occupancy_frac (environment only)
    - strategy_pass_frac (how often Greg passes this strategy in each cell)
    """
    samples_list = list(samples)
    
    x_vals = []
    y_vals = []
    sample_data = []
    
    for s in samples_list:
        xv = s.get(x_metric)
        yv = s.get(y_metric)
        if xv is not None and yv is not None:
            x_vals.append(xv)
            y_vals.append(yv)
            sample_data.append(s)
    
    if not x_vals:
        return []
    
    if x_range is None:
        x_min, x_max = min(x_vals), max(x_vals)
        margin = (x_max - x_min) * 0.05 if x_max > x_min else 1.0
        x_range = (x_min - margin, x_max + margin)
    
    if y_range is None:
        y_min, y_max = min(y_vals), max(y_vals)
        margin = (y_max - y_min) * 0.05 if y_max > y_min else 1.0
        y_range = (y_min - margin, y_max + margin)
    
    x_step = (x_range[1] - x_range[0]) / x_bins
    y_step = (y_range[1] - y_range[0]) / y_bins
    
    counts = [[0 for _ in range(x_bins)] for _ in range(y_bins)]
    strategy_counts = [[0 for _ in range(x_bins)] for _ in range(y_bins)]
    total = 0
    
    for xv, yv, sample in zip(x_vals, y_vals, sample_data):
        ix = int((xv - x_range[0]) / x_step) if x_step > 0 else 0
        iy = int((yv - y_range[0]) / y_step) if y_step > 0 else 0
        
        ix = max(0, min(x_bins - 1, ix))
        iy = max(0, min(y_bins - 1, iy))
        
        counts[iy][ix] += 1
        total += 1
        
        if _check_strategy_passes(sample, strategy_key, {}):
            strategy_counts[iy][ix] += 1
    
    cells = []
    for iy in range(y_bins):
        for ix in range(x_bins):
            x_low = x_range[0] + ix * x_step
            x_high = x_low + x_step
            y_low = y_range[0] + iy * y_step
            y_high = y_low + y_step
            count = counts[iy][ix]
            strat_count = strategy_counts[iy][ix]
            occupancy_frac = count / total if total > 0 else 0.0
            pass_frac = strat_count / count if count > 0 else 0.0
            
            cells.append(StrategyHeatmapCell(
                x_low=round(x_low, 4),
                x_high=round(x_high, 4),
                y_low=round(y_low, 4),
                y_high=round(y_high, 4),
                count=count,
                occupancy_frac=round(occupancy_frac, 6),
                total_samples_in_cell=count,
                strategy_selected_count=strat_count,
                strategy_pass_frac=round(pass_frac, 6),
            ))
    
    return cells


def save_heatmap_csv(
    cells: List[HeatmapCell],
    output_path: Path,
    x_metric: str,
    y_metric: str,
    underlying: str,
) -> None:
    """
    Save a heatmap grid to CSV with headers:
    underlying, x_metric, y_metric, x_low, x_high, y_low, y_high, count, occupancy_frac
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "underlying", "x_metric", "y_metric",
            "x_low", "x_high", "y_low", "y_high",
            "count", "occupancy_frac"
        ])
        for cell in cells:
            writer.writerow([
                underlying, x_metric, y_metric,
                cell.x_low, cell.x_high, cell.y_low, cell.y_high,
                cell.count, cell.occupancy_frac
            ])


def save_strategy_heatmap_csv(
    cells: List[StrategyHeatmapCell],
    output_path: Path,
    x_metric: str,
    y_metric: str,
    underlying: str,
    strategy_key: str,
) -> None:
    """Save a strategy heatmap grid to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "underlying", "strategy", "x_metric", "y_metric",
            "x_low", "x_high", "y_low", "y_high",
            "count", "occupancy_frac", "strategy_pass_frac"
        ])
        for cell in cells:
            writer.writerow([
                underlying, strategy_key, x_metric, y_metric,
                cell.x_low, cell.x_high, cell.y_low, cell.y_high,
                cell.count, cell.occupancy_frac, cell.strategy_pass_frac
            ])


def find_sweet_spots(
    cells: List[StrategyHeatmapCell],
    min_env_frac: float = 0.005,
    min_pass_frac: float = 0.4,
    top_k: int = 5,
) -> List[SweetSpot]:
    """
    Find sweet spot regions where:
    - Environment spends non-trivial time (occupancy_frac >= min_env_frac)
    - Strategy passes frequently (strategy_pass_frac >= min_pass_frac)
    
    Returns top K by sweetness = occupancy_frac * strategy_pass_frac
    """
    candidates = []
    
    for cell in cells:
        if cell.occupancy_frac >= min_env_frac and cell.strategy_pass_frac >= min_pass_frac:
            sweetness = cell.occupancy_frac * cell.strategy_pass_frac
            candidates.append(SweetSpot(
                x_low=cell.x_low,
                x_high=cell.x_high,
                y_low=cell.y_low,
                y_high=cell.y_high,
                occupancy_frac=cell.occupancy_frac,
                strategy_pass_frac=cell.strategy_pass_frac,
                sweetness=round(sweetness, 6),
            ))
    
    candidates.sort(key=lambda x: x.sweetness, reverse=True)
    return candidates[:top_k]


def run_full_heatmap_analysis(
    underlyings: List[str],
    metrics: List[str],
    strategies: List[str],
    x_bins: int = 20,
    y_bins: int = 20,
    min_env_frac: float = 0.005,
    min_pass_frac: float = 0.4,
    lookback_days: int = 365,
    output_dir: Path = Path("backtest/output"),
) -> List[StrategySweetSpotSummary]:
    """
    Run full heatmap analysis for all metric pairs and strategies.
    
    Args:
        underlyings: List of underlying assets (e.g., ["BTC", "ETH"])
        metrics: List of metric names to analyze
        strategies: List of strategy labels or keys
        x_bins, y_bins: Number of bins per axis
        min_env_frac: Minimum occupancy fraction for sweet spots
        min_pass_frac: Minimum strategy pass rate for sweet spots
        lookback_days: Historical lookback period
        output_dir: Output directory for CSV/JSON/Markdown files
    
    Returns:
        List of StrategySweetSpotSummary objects
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    all_summaries = []
    
    strategy_keys = []
    for s in strategies:
        if s in LABEL_TO_KEY:
            strategy_keys.append(LABEL_TO_KEY[s])
        elif s in GREG_STRATEGY_CRITERIA:
            strategy_keys.append(s)
        else:
            print(f"Warning: Unknown strategy '{s}', skipping")
    
    for underlying in underlyings:
        print(f"\n=== Processing {underlying} ===")
        
        samples = list(load_metric_samples_for_underlying(
            underlying, lookback_days=lookback_days
        ))
        print(f"Loaded {len(samples)} samples")
        
        metric_pairs = []
        for i, m1 in enumerate(metrics):
            for m2 in metrics[i+1:]:
                metric_pairs.append((m1, m2))
        
        for x_metric, y_metric in metric_pairs:
            print(f"  Computing environment heatmap: {x_metric} vs {y_metric}")
            
            env_cells = compute_environment_heatmap(
                samples, x_metric, y_metric, x_bins, y_bins
            )
            
            csv_path = output_dir / f"env_heatmap_{underlying}_{x_metric}_vs_{y_metric}.csv"
            save_heatmap_csv(env_cells, csv_path, x_metric, y_metric, underlying)
            
            for strategy_key in strategy_keys:
                strategy_label = STRATEGY_LABELS.get(strategy_key, strategy_key)
                print(f"    Strategy: {strategy_label}")
                
                strat_cells = compute_strategy_heatmap_for_pair(
                    samples, x_metric, y_metric, strategy_key, underlying,
                    x_bins, y_bins
                )
                
                sweet_spots = find_sweet_spots(
                    strat_cells, min_env_frac, min_pass_frac
                )
                
                if sweet_spots:
                    summary = StrategySweetSpotSummary(
                        underlying=underlying,
                        strategy=strategy_label,
                        strategy_key=strategy_key,
                        x_metric=x_metric,
                        y_metric=y_metric,
                        sweet_spots=sweet_spots,
                    )
                    all_summaries.append(summary)
    
    return all_summaries


def save_sweetspots_json(summaries: List[StrategySweetSpotSummary], output_path: Path) -> None:
    """Save sweet spots to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    data = []
    for s in summaries:
        data.append({
            "underlying": s.underlying,
            "strategy": s.strategy,
            "strategy_key": s.strategy_key,
            "x_metric": s.x_metric,
            "y_metric": s.y_metric,
            "sweet_spots": [asdict(sp) for sp in s.sweet_spots],
        })
    
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


def save_sweetspots_markdown(
    summaries: List[StrategySweetSpotSummary],
    output_path: Path,
    min_env_frac: float,
    min_pass_frac: float,
) -> None:
    """Save sweet spots to human-readable Markdown file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    by_underlying: Dict[str, List[StrategySweetSpotSummary]] = {}
    for s in summaries:
        by_underlying.setdefault(s.underlying, []).append(s)
    
    lines = ["# Greg Environment Sweet Spots\n"]
    lines.append(f"Generated from synthetic universe analysis.\n")
    lines.append(f"- Minimum environment occupancy: {min_env_frac * 100:.1f}%\n")
    lines.append(f"- Minimum strategy pass rate: {min_pass_frac * 100:.0f}%\n\n")
    
    for underlying in sorted(by_underlying.keys()):
        lines.append(f"# {underlying}\n\n")
        
        summaries_u = by_underlying[underlying]
        by_strategy: Dict[str, List[StrategySweetSpotSummary]] = {}
        for s in summaries_u:
            by_strategy.setdefault(s.strategy, []).append(s)
        
        for strategy in sorted(by_strategy.keys()):
            summaries_s = by_strategy[strategy]
            lines.append(f"## {strategy}\n\n")
            
            for s in summaries_s:
                lines.append(f"### ({s.x_metric}, {s.y_metric})\n\n")
                
                if not s.sweet_spots:
                    lines.append("No sweet spots found.\n\n")
                    continue
                
                lines.append("| Rank | X Range | Y Range | Occupancy | Pass Rate | Score |\n")
                lines.append("|------|---------|---------|-----------|-----------|-------|\n")
                
                for i, sp in enumerate(s.sweet_spots, 1):
                    x_range = f"{sp.x_low:.1f} - {sp.x_high:.1f}"
                    y_range = f"{sp.y_low:.1f} - {sp.y_high:.1f}"
                    occ = f"{sp.occupancy_frac * 100:.1f}%"
                    pass_r = f"{sp.strategy_pass_frac * 100:.0f}%"
                    score = f"{sp.sweetness:.4f}"
                    lines.append(f"| {i} | {x_range} | {y_range} | {occ} | {pass_r} | {score} |\n")
                
                lines.append("\n")
    
    with open(output_path, "w") as f:
        f.writelines(lines)
