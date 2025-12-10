"""
Greg Sweet Spots Sweep - Automated environment analysis.

Provides a single function to run the full Greg environment sweet spot sweep
from the UI without requiring CLI invocation.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

from src.backtest.environment_heatmap import (
    run_full_heatmap_analysis,
    save_sweetspots_json,
    save_sweetspots_markdown,
    AVAILABLE_METRICS,
    GREG_STRATEGIES,
)

DEFAULT_UNDERLYINGS: Sequence[str] = ("BTC", "ETH")

DEFAULT_METRICS: Sequence[str] = [
    "vrp_30d",
    "adx_14d",
    "iv_rank_6m",
    "skew_25d",
    "term_structure_spread",
    "chop_factor_7d",
    "price_vs_ma200",
    "rsi_14d",
]

DEFAULT_STRATEGIES: Sequence[str] = [
    "STRATEGY_A_STRADDLE",
    "STRATEGY_A_STRANGLE",
    "STRATEGY_B_CALENDAR",
    "STRATEGY_C_SHORT_PUT",
    "STRATEGY_D_IRON_BUTTERFLY",
    "STRATEGY_F_BULL_PUT_SPREAD",
    "STRATEGY_F_BEAR_CALL_SPREAD",
]


def run_greg_sweetspot_sweep(
    base_dir: Path,
    underlyings: Sequence[str] = DEFAULT_UNDERLYINGS,
    metrics: Sequence[str] = DEFAULT_METRICS,
    strategies: Sequence[str] = DEFAULT_STRATEGIES,
    x_bins: int = 10,
    y_bins: int = 10,
    min_env_frac: float = 0.005,
    min_pass_frac: float = 0.4,
    lookback_days: int = 90,
    top_k: int = 5,
) -> Path:
    """
    Run a fixed Greg environment sweep across a curated list of metric pairs
    and strategies, then save:

    - Env heatmap CSVs under backtest/output/
    - Sweet spot summary JSON under backtest/output/greg_heatmap_sweetspots.json
    - Sweet spot summary Markdown under backtest/output/greg_heatmap_sweetspots.md

    Args:
        base_dir: Project base directory (parent of backtest/)
        underlyings: List of underlying assets to analyze
        metrics: List of metric names to analyze
        strategies: List of Greg strategy keys
        x_bins, y_bins: Number of bins per axis (smaller for faster runs)
        min_env_frac: Minimum environment occupancy fraction (0.5% default)
        min_pass_frac: Minimum strategy pass rate (40% default)
        lookback_days: Historical lookback period
        top_k: Number of top sweet spots to keep per strategy/pair

    Returns:
        Path to the generated greg_heatmap_sweetspots.json file.
    """
    output_dir = base_dir / "backtest" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = run_full_heatmap_analysis(
        underlyings=list(underlyings),
        metrics=list(metrics),
        strategies=list(strategies),
        x_bins=x_bins,
        y_bins=y_bins,
        min_env_frac=min_env_frac,
        min_pass_frac=min_pass_frac,
        lookback_days=lookback_days,
        output_dir=output_dir,
    )

    json_path = output_dir / "greg_heatmap_sweetspots.json"
    save_sweetspots_json(summaries, json_path)

    md_path = output_dir / "greg_heatmap_sweetspots.md"
    save_sweetspots_markdown(summaries, md_path, min_env_frac, min_pass_frac)

    return json_path
