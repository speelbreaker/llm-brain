#!/usr/bin/env python3
"""Run the Strategy PnL Parity + Synthetic Fidelity suite.

P0: Produces a deterministic report with wiring + score skeleton.

Usage:
  python -m scripts.run_fidelity_suite --underlying BTC --start 2025-12-01 --end 2025-12-05

Outputs (in --out-dir):
  - fidelity_report.json
  - fidelity_report.md

Next iterations will populate real strategy execution + parity metrics.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List

from src.backtest.types import CallSimulationConfig
from src.fidelity.market_replay import LiveReplayMarket, SyntheticReplayMarket, FillModelConfig
from src.fidelity.parity_runner import run_parity_suite
from src.fidelity.reporting import FidelityReport, now_iso, write_report_json, write_report_md
from src.fidelity.scoring import score_components, gate_label
from src.fidelity.strategies_canonical import canonical_strategy_specs


def _parse_date(s: str) -> datetime:
    # Interpret as UTC midnight.
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _decision_times(start: datetime, end: datetime) -> List[datetime]:
    # P0: daily decisions at 00:00 UTC.
    t = start
    out: List[datetime] = []
    while t <= end:
        out.append(t)
        t = t + timedelta(days=1)
    return out


def _has_any_harvested_snapshots(*, base_dir: str, underlying_dir: str, start: datetime, end: datetime) -> bool:
    """Best-effort preflight check for harvested snapshot availability.

    The exam dataset is stored under:
      {base_dir}/{UNDERLYING_DIR}/YYYY/MM/DD/*.parquet

    We check day folders in [start.date(), end.date()] and look for at least one parquet file.
    """
    root = Path(base_dir) / underlying_dir
    if not root.exists():
        return False

    d = start.date()
    end_d = end.date()
    while d <= end_d:
        day_dir = root / f"{d.year:04d}" / f"{d.month:02d}" / f"{d.day:02d}"
        if day_dir.exists():
            try:
                if any(p.suffix == ".parquet" for p in day_dir.iterdir()):
                    return True
            except Exception:
                pass
        d = d + timedelta(days=1)

    return False


def main() -> None:
    p = argparse.ArgumentParser(description="Run synthetic fidelity suite")
    p.add_argument("--underlying", required=True, choices=["BTC", "ETH"], help="Underlying")
    p.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    p.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    p.add_argument("--out-dir", default=None, help="Output directory (defaults to data/fidelity_runs/<UNDERLYING>/latest)")
    p.add_argument(
        "--live-data-dir",
        default="data/live_deribit",
        help="Base directory for harvested Deribit snapshot data (exam dataset)",
    )
    p.add_argument("--fill-bps", type=float, default=0.0, help="Fixed fill bps (Phase 1)")

    args = p.parse_args()

    underlying = args.underlying
    start = _parse_date(args.start)
    end = _parse_date(args.end)

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = Path("data/fidelity_runs") / underlying / "latest"
    out_dir.mkdir(parents=True, exist_ok=True)

    decision_times = _decision_times(start, end)

    # DataSource wiring (P0): rely on existing backtest DS implementations.
    # Live replay should be backed by harvested data source (LiveDeribitDataSource).
    from src.backtest.live_deribit_data_source import LiveDeribitDataSource

    underlying_dir = f"{underlying}_USDC"

    if not _has_any_harvested_snapshots(
        base_dir=args.live_data_dir,
        underlying_dir=underlying_dir,
        start=start,
        end=end,
    ):
        print(
            "No harvested snapshot parquet files found for the requested window.\n"
            f"- underlying_dir: {underlying_dir}\n"
            f"- window: {start.date().isoformat()} to {end.date().isoformat()}\n"
            f"- base_dir: {args.live_data_dir}\n\n"
            "Pick a date range that exists under the dataset folders, e.g.\n"
            f"  {args.live_data_dir}/{underlying_dir}/YYYY/MM/DD/*.parquet\n"
        )
        raise SystemExit(2)

    live_ds = LiveDeribitDataSource(
        underlying=underlying_dir,
        start_date=start.date(),
        end_date=end.date(),
        base_dir=args.live_data_dir,
        canonical_underlying=underlying,
    )

    # Synthetic market uses build_historical_state(); needs a CallSimulationConfig.
    # Keep config minimal + safe defaults.
    cfg = CallSimulationConfig(
        underlying=underlying,
        start=start,
        end=end,
        timeframe="1d",
        decision_interval_bars=1,
        initial_spot_position=1.0,
        contract_size=1.0,
        fee_rate=0.0005,
    )
    cfg = replace(cfg, chain_mode="synthetic_grid", pricing_mode="synthetic_bs")

    fill_cfg = FillModelConfig(bps=float(args.fill_bps))

    live_market = LiveReplayMarket(
        underlying=underlying,
        ds=live_ds,
        fill_cfg=fill_cfg,
    )
    synthetic_market = SyntheticReplayMarket(
        underlying=underlying,
        ds=live_ds,
        cfg=cfg,
        fill_cfg=fill_cfg,
    )

    strategies = canonical_strategy_specs()
    parity = run_parity_suite(
        decision_times=decision_times,
        strategies=strategies,
        live_market=live_market,
        synthetic_market=synthetic_market,
    )

    # P0 scoring skeleton: provide deterministic placeholders.
    components = {
        "underlying_returns": {
            "weight": 0.20,
            "metrics": {
                "placeholder": {"error": 0.0, "tolerance": 1.0, "k": 1.0, "weight": 1.0},
            },
        },
        "strategy_pnl_parity": {
            "weight": 0.20,
            "metrics": {
                "placeholder": {"error": 0.0, "tolerance": 1.0, "k": 1.0, "weight": 1.0},
            },
        },
    }
    scored = score_components(components)
    overall = float(scored["overall_score"])

    # In P0, parity metrics are placeholders; set derived scores accordingly.
    strategy_parity_score = float(scored["component_scores"].get("strategy_pnl_parity", 0.0))
    tail_parity_score = float(scored["component_scores"].get("strategy_pnl_parity", 0.0))

    gate = gate_label(
        overall_score=overall,
        strategy_parity_score=strategy_parity_score,
        tail_parity_score=tail_parity_score,
    )

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    history_dir = Path("data/fidelity_runs") / underlying / "history" / run_id
    history_dir.mkdir(parents=True, exist_ok=True)

    report = FidelityReport(
        run_id=run_id,
        timestamp=now_iso(),
        market_live_meta=live_market.meta(),
        market_synth_meta=synthetic_market.meta(),
        component_scores=scored["component_scores"],
        overall_score=overall,
        gate=gate,
        strategy_parity=parity,
    )

    write_report_json(report, out_dir / "fidelity_report.json")
    write_report_md(report, out_dir / "fidelity_report.md")

    # Also persist a versioned copy for API history.
    write_report_json(report, history_dir / "fidelity_report.json")
    write_report_md(report, history_dir / "fidelity_report.md")

    print(f"Wrote {out_dir / 'fidelity_report.json'}")
    print(f"Wrote {out_dir / 'fidelity_report.md'}")
    print(f"Wrote {history_dir / 'fidelity_report.json'}")
    print(f"Wrote {history_dir / 'fidelity_report.md'}")


if __name__ == "__main__":
    main()
