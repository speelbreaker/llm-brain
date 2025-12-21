"""Reusable comparison logic for SYNTHETIC vs LIVE_DERIBIT backtests.

This module intentionally reuses the same simulator and file-based storage
conventions as the Backtest Lab (see src/backtest/manager.py + src/backtest/run_store.py).

It is DB-free by design so it can run in environments without DATABASE_URL.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Tuple

from src.backtest.config_schema import DataSourceType
from src.backtest.covered_call_simulator import CoveredCallSimulator
from src.backtest.deribit_data_source import DeribitDataSource
from src.backtest.run_store import create_run, save_run_result, update_run_status
from src.backtest.types import CallSimulationConfig


def run_backtest_with_data_source(
    underlying: str,
    start_ts: datetime,
    end_ts: datetime,
    data_source: DataSourceType,
    decision_interval_minutes: int,
    exit_style: str,
    verbose: bool = True,
) -> str:
    """
    Run a backtest and return the run_id.
    
    Args:
        underlying: Asset to backtest (e.g., "BTC", "ETH")
        start_ts: Start timestamp (UTC)
        end_ts: End timestamp (UTC)
        data_source: SYNTHETIC or LIVE_DERIBIT
        decision_interval_minutes: Decision interval in minutes
        exit_style: Exit style (e.g., "tp_and_roll")
        verbose: Whether to print progress messages
        
    Returns:
        run_id string
        
    Raises:
        Exception if backtest fails
    """
    run_result = create_run(
        {
            "underlying": underlying,
            "start_date": start_ts.date().isoformat(),
            "end_date": end_ts.date().isoformat(),
            "decision_interval_minutes": decision_interval_minutes,
            "exit_style": exit_style,
            "data_source": data_source.value,
        }
    )
    run_id = run_result.run_id
    update_run_status(run_id, "running")

    if verbose:
        print(f"  Created run: {run_id} (data_source={data_source.value})")
            
    try:
        decision_interval_hours = decision_interval_minutes / 60
        decision_interval_bars = max(1, int(decision_interval_hours))

        pricing_mode = "deribit_live" if data_source == DataSourceType.LIVE_DERIBIT else "synthetic_bs"

        # For small harvested windows (e.g. a few days), a 7DTE strategy produces
        # zero decision points because the simulator stops at end - target_dte.
        # Adapt target_dte to the available window while keeping it bounded.
        window_days = max(1, int((end_ts - start_ts).total_seconds() / 86400))
        target_dte = max(1, min(7, window_days // 2))
        dte_tolerance = max(1, min(3, target_dte))

        config = CallSimulationConfig(
            underlying=underlying,
            start=start_ts,
            end=end_ts,
            timeframe="1h",
            decision_interval_bars=decision_interval_bars,
            initial_spot_position=1.0,
            contract_size=1.0,
            fee_rate=0.0003,
            target_dte=target_dte,
            dte_tolerance=dte_tolerance,
            target_delta=0.25,
            delta_tolerance=0.10,
            min_dte=1,
            max_dte=21,
            delta_min=0.10,
            delta_max=0.40,
            option_margin_type="linear",
            option_settlement_ccy="USDC",
            tp_threshold_pct=80.0,
            # Fidelity wants measurability; we intentionally trade whenever we have candidates.
            min_score_to_trade=0.0,
            pricing_mode=pricing_mode,
            # Align synthetic runs to the harvested chain universe when available.
            chain_mode="live_chain",
            sigma_mode="mark_iv_x_multiplier",
            synthetic_iv_multiplier=1.0,
        )

        if data_source in (DataSourceType.LIVE_DERIBIT, DataSourceType.SYNTHETIC):
            from src.backtest.live_deribit_data_source import LiveDeribitDataSource

            underlying_dir = underlying if "_USDC" in underlying else f"{underlying}_USDC"
            data_src = LiveDeribitDataSource(
                underlying=underlying_dir,
                start_date=start_ts.date(),
                end_date=end_ts.date(),
                canonical_underlying=underlying,
            )
        else:
            data_src = DeribitDataSource()

        simulator = CoveredCallSimulator(data_source=data_src, config=config)

        from src.backtest.state_builder import build_historical_state

        # Prefer decision times aligned to harvested snapshots (when available) so
        # list_option_chain/get_option_ohlc have data and we don't rely on fallbacks.
        decision_times = simulator._generate_decision_times()
        if hasattr(data_src, "get_dataframe"):
            try:
                import pandas as pd  # local import to keep module load light

                df = data_src.get_dataframe()
                if df is not None and (not df.empty) and "harvest_time" in df.columns:
                    cutoff = end_ts - timedelta(days=target_dte)
                    raw_times = sorted(pd.to_datetime(df["harvest_time"], utc=True).unique())
                    snap_times: list[datetime] = []
                    for ts in raw_times:
                        if hasattr(ts, "to_pydatetime"):
                            snap_times.append(ts.to_pydatetime())
                        else:
                            snap_times.append(ts)
                    snap_times = [t for t in snap_times if start_ts <= t <= cutoff]
                    # Downsample to roughly the requested decision interval.
                    selected: list[datetime] = []
                    last: Optional[datetime] = None
                    min_step = int(decision_interval_minutes) * 60
                    for t in snap_times:
                        if last is None or (t - last).total_seconds() >= min_step:
                            selected.append(t)
                            last = t
                    if selected:
                        decision_times = selected
            except Exception:
                # Fall back to regular time grid.
                pass

        def state_builder(t: datetime):
            try:
                return build_historical_state(data_src, config, t)
            except Exception as e:
                # For fidelity runs we prefer "skip this decision point" over
                # aborting the entire compare run.
                spot_df = data_src.get_spot_ohlc(
                    underlying=underlying,
                    start=t - timedelta(hours=24),
                    end=t,
                    timeframe="1h",
                )
                spot = float(spot_df["close"].iloc[-1]) if not spot_df.empty else None
                return {
                    "time": t,
                    "spot": spot,
                    "underlying": underlying,
                    "market_context": {},
                    "candidate_options": [],
                    "portfolio": {"spot_position": config.initial_spot_position, "equity_usd": None},
                    "provenance": {"error": str(e)},
                }

        result = simulator.simulate_policy_with_scoring(
            decision_times=decision_times,
            state_builder=state_builder,
            exit_style=exit_style,
            min_score_to_trade=config.min_score_to_trade,
            size=1.0,
        )

        trades = result.trades if hasattr(result, "trades") else []
        metrics = result.metrics if hasattr(result, "metrics") else {}

        chains_list = []
        for trade in trades:
            chain = getattr(trade, "chain", None)
            if chain:
                chains_list.append(
                    {
                        "open_time": chain.decision_time.isoformat(),
                        "instrument_name": getattr(chain, "instrument_name", None),
                        "num_legs": len(getattr(chain, "legs", [])),
                        "num_rolls": max(0, len(getattr(chain, "legs", [])) - 1),
                        "pnl": float(chain.total_pnl),
                        "pnl_vs_hodl": float(getattr(chain, "pnl_vs_hodl", 0)),
                        "max_drawdown_pct": float(chain.max_drawdown_pct),
                    }
                )

        num_trades = int(metrics.get("num_trades", 0) or 0)
        net_profit_usd = float(metrics.get("final_pnl", 0.0) or 0.0)
        gross_profit = float(sum(float(t.pnl) for t in trades if float(t.pnl) > 0.0))
        gross_loss = float(abs(sum(float(t.pnl) for t in trades if float(t.pnl) < 0.0)))
        if gross_loss > 0:
            profit_factor = gross_profit / gross_loss
        else:
            profit_factor = 10.0 if gross_profit > 0 else 0.0
        avg_trade_usd = (net_profit_usd / num_trades) if num_trades > 0 else 0.0
        final_pnl_vs_hodl = float(sum(float(getattr(t, "pnl_vs_hodl", 0.0) or 0.0) for t in trades))

        initial_equity = float(getattr(config, "initial_capital_usd", 0.0) or 0.0)
        final_equity = initial_equity + net_profit_usd
        net_profit_pct = (net_profit_usd / initial_equity * 100.0) if initial_equity > 0 else 0.0
        max_dd_pct = float(metrics.get("max_drawdown_pct", 0.0) or 0.0)

        formatted_metrics = {
            "initial_equity": initial_equity,
            "final_equity": final_equity,
            "net_profit_usd": net_profit_usd,
            "net_profit_pct": net_profit_pct,
            "final_pnl_vs_hodl": final_pnl_vs_hodl,
            "max_drawdown_pct": max_dd_pct,
            "max_drawdown_usd": (max_dd_pct / 100.0) * initial_equity if initial_equity > 0 else 0.0,
            "num_trades": num_trades,
            "win_rate": float(metrics.get("win_rate", 0.0) or 0.0) * 100.0,
            "profit_factor": float(profit_factor),
            "avg_trade_usd": float(avg_trade_usd),
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
        }

        run_result.status = "finished"
        run_result.metrics = {exit_style: formatted_metrics}
        run_result.recent_chains = {exit_style: chains_list}
        save_run_result(run_result)
        update_run_status(run_id, "finished")

        if hasattr(data_src, "close"):
            data_src.close()

        if verbose:
            print(f"  Completed run: {run_id}")
        return run_id

    except Exception as e:
        update_run_status(run_id, "failed", error=str(e))
        raise


def run_synthetic_vs_live_pair(
    underlying: str,
    start_ts: datetime,
    end_ts: datetime,
    decision_interval_minutes: int,
    exit_style: str,
    verbose: bool = True,
) -> Tuple[str, str]:
    """
    Run a pair of backtests (SYNTHETIC and LIVE_DERIBIT) and return both run_ids.
    
    Args:
        underlying: Asset to backtest
        start_ts: Start timestamp (UTC)
        end_ts: End timestamp (UTC)
        decision_interval_minutes: Decision interval in minutes
        exit_style: Exit style
        verbose: Whether to print progress messages
        
    Returns:
        Tuple of (synthetic_run_id, live_deribit_run_id)
        
    Raises:
        Exception if either backtest fails
    """
    if verbose:
        print(f"Running SYNTHETIC backtest for {underlying}...")
    synth_run_id = run_backtest_with_data_source(
        underlying=underlying,
        start_ts=start_ts,
        end_ts=end_ts,
        data_source=DataSourceType.SYNTHETIC,
        decision_interval_minutes=decision_interval_minutes,
        exit_style=exit_style,
        verbose=verbose,
    )
    
    if verbose:
        print(f"Running LIVE_DERIBIT backtest for {underlying}...")
    live_run_id = run_backtest_with_data_source(
        underlying=underlying,
        start_ts=start_ts,
        end_ts=end_ts,
        data_source=DataSourceType.LIVE_DERIBIT,
        decision_interval_minutes=decision_interval_minutes,
        exit_style=exit_style,
        verbose=verbose,
    )
    
    return synth_run_id, live_run_id


def get_metrics_for_run(run_id: str, exit_style: str) -> Optional[Dict[str, Any]]:
    """Get metrics for a completed run."""
    from src.backtest.run_store import load_result

    result = load_result(run_id)
    if not result:
        return None
    return (result.metrics or {}).get(exit_style, {})
