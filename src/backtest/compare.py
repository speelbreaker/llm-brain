"""
Reusable comparison logic for SYNTHETIC vs LIVE_DERIBIT backtests.
"""

from datetime import datetime
from typing import Tuple, Optional, Dict, Any

from src.db import get_db_session
from src.db.backtest_service import (
    create_backtest_run,
    complete_run,
    fail_run,
    get_run_with_details,
)
from src.backtest.config_schema import DataSourceType
from src.backtest.covered_call_simulator import CoveredCallSimulator
from src.backtest.deribit_data_source import DeribitDataSource
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
    with get_db_session() as db:
        try:
            run = create_backtest_run(
                db=db,
                underlying=underlying,
                start_ts=start_ts,
                end_ts=end_ts,
                data_source=data_source.value,
                decision_interval_minutes=decision_interval_minutes,
                primary_exit_style=exit_style,
                config_json={
                    "underlying": underlying,
                    "data_source": data_source.value,
                    "decision_interval_minutes": decision_interval_minutes,
                    "exit_style": exit_style,
                },
            )
            
            if verbose:
                print(f"  Created run: {run.run_id} (data_source={data_source.value})")
            
            decision_interval_hours = decision_interval_minutes / 60
            decision_interval_bars = max(1, int(decision_interval_hours))
            
            pricing_mode = "deribit_live" if data_source == DataSourceType.LIVE_DERIBIT else "synthetic_bs"
            
            config = CallSimulationConfig(
                underlying=underlying,
                start=start_ts,
                end=end_ts,
                timeframe="1h",
                decision_interval_bars=decision_interval_bars,
                initial_spot_position=1.0,
                contract_size=1.0,
                fee_rate=0.0003,
                target_dte=7,
                dte_tolerance=3,
                target_delta=0.25,
                delta_tolerance=0.10,
                min_dte=1,
                max_dte=21,
                delta_min=0.10,
                delta_max=0.40,
                option_margin_type="linear",
                option_settlement_ccy="USDC",
                tp_threshold_pct=80.0,
                min_score_to_trade=3.0,
                pricing_mode=pricing_mode,
            )
            
            if data_source == DataSourceType.LIVE_DERIBIT:
                from src.backtest.live_deribit_data_source import LiveDeribitDataSource
                
                data_src = LiveDeribitDataSource(
                    underlying=underlying,
                    start_date=start_ts.date(),
                    end_date=end_ts.date(),
                )
            else:
                data_src = DeribitDataSource()
            
            simulator = CoveredCallSimulator(data_source=data_src, config=config)
            
            def always_trade_policy(candidates, state):
                return True
            
            result = simulator.simulate_policy(policy=always_trade_policy, size=1.0)
            
            trades = result.trades if hasattr(result, 'trades') else []
            metrics = result.metrics if hasattr(result, 'metrics') else {}
            
            chains_list = []
            for trade in trades:
                chain = getattr(trade, "chain", None)
                if chain:
                    chains_list.append({
                        "open_time": chain.decision_time.isoformat(),
                        "instrument_name": getattr(chain, "instrument_name", None),
                        "num_legs": len(getattr(chain, "legs", [])),
                        "num_rolls": max(0, len(getattr(chain, "legs", [])) - 1),
                        "pnl": float(chain.total_pnl),
                        "pnl_vs_hodl": float(getattr(chain, "pnl_vs_hodl", 0)),
                        "max_drawdown_pct": float(chain.max_drawdown_pct),
                    })
            
            formatted_metrics = {
                "initial_equity": metrics.get("initial_equity", 0),
                "final_equity": metrics.get("final_equity", 0),
                "net_profit_usd": metrics.get("final_pnl", 0),
                "net_profit_pct": metrics.get("total_return_pct", 0),
                "max_drawdown_pct": metrics.get("max_drawdown_pct", 0),
                "num_trades": metrics.get("num_trades", 0),
                "win_rate": metrics.get("win_rate", 0) * 100 if metrics.get("win_rate") else 0,
                "sharpe_ratio": metrics.get("sharpe_ratio", 0),
                "sortino_ratio": metrics.get("sortino_ratio", 0),
                "profit_factor": metrics.get("profit_factor", 0),
                "final_pnl_vs_hodl": metrics.get("total_pnl_vs_hodl", 0),
            }
            
            metrics_by_style = {exit_style: formatted_metrics}
            chains_by_style = {exit_style: chains_list}
            
            complete_run(
                db=db,
                run=run,
                metrics_by_style=metrics_by_style,
                chains_by_style=chains_by_style,
                primary_exit_style=exit_style,
            )
            
            if hasattr(data_src, 'close'):
                data_src.close()
            
            if verbose:
                print(f"  Completed run: {run.run_id}")
            return run.run_id
            
        except Exception as e:
            if 'run' in locals():
                fail_run(db, run, str(e))
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
    with get_db_session() as db:
        result = get_run_with_details(db, run_id)
        if not result:
            return None
        
        metrics = result.get("metrics", {})
        return metrics.get(exit_style, {})
