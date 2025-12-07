#!/usr/bin/env python3
"""
Run Backtest with REAL_SCRAPER Data Source

Example script to run a backtest using Real Scraper data (e.g., 2023-05-26 BTC).
Results are persisted to PostgreSQL using the same code path as the FastAPI endpoint.

Usage:
    python scripts/run_backtest_real_scraper_example.py \
        --underlying BTC \
        --date 2023-05-26
"""

import argparse
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def main():
    parser = argparse.ArgumentParser(
        description="Run backtest with REAL_SCRAPER data source"
    )
    parser.add_argument(
        "--underlying",
        default="BTC",
        help="Underlying asset (default: BTC)",
    )
    parser.add_argument(
        "--date",
        default="2023-05-26",
        help="Date to backtest YYYY-MM-DD (default: 2023-05-26)",
    )
    parser.add_argument(
        "--decision-interval",
        type=int,
        default=60,
        help="Decision interval in minutes (default: 60)",
    )
    parser.add_argument(
        "--target-delta",
        type=float,
        default=0.25,
        help="Target delta for covered calls (default: 0.25)",
    )
    parser.add_argument(
        "--target-dte",
        type=int,
        default=7,
        help="Target DTE in days (default: 7)",
    )
    
    args = parser.parse_args()
    
    underlying = args.underlying.upper()
    target_date = args.date
    
    try:
        date_dt = datetime.strptime(target_date, "%Y-%m-%d")
    except ValueError:
        print(f"ERROR: Invalid date format: {target_date}")
        sys.exit(1)
    
    data_path = Path(f"data/real_scraper/{underlying}/{target_date}/{underlying}_{target_date}.parquet")
    if not data_path.exists():
        print(f"ERROR: Real Scraper data not found at: {data_path}")
        print(f"\nPlease import the data first using:")
        print(f"  python scripts/import_real_scraper_deribit.py --underlying {underlying} --date {target_date} --input <path-to-csv>")
        sys.exit(1)
    
    print("=" * 60)
    print("REAL_SCRAPER BACKTEST")
    print("=" * 60)
    print(f"Underlying:         {underlying}")
    print(f"Date:               {target_date}")
    print(f"Decision interval:  {args.decision_interval} minutes")
    print(f"Target delta:       {args.target_delta}")
    print(f"Target DTE:         {args.target_dte} days")
    print(f"Data file:          {data_path}")
    print("=" * 60)
    
    from src.backtest.types import CallSimulationConfig
    from src.backtest.data_source import Timeframe
    from src.backtest.covered_call_simulator import CoveredCallSimulator, always_trade_policy
    from src.backtest.real_scraper_data_source import RealScraperDataSource
    from src.db import get_db_session, init_db
    from src.db.backtest_service import create_backtest_run, complete_run, fail_run
    
    init_db()
    
    start_ts = datetime(date_dt.year, date_dt.month, date_dt.day, 0, 0, 0, tzinfo=timezone.utc)
    end_ts = datetime(date_dt.year, date_dt.month, date_dt.day, 23, 59, 59, tzinfo=timezone.utc)
    
    timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
    run_id = f"{timestamp}_{underlying}_{uuid.uuid4().hex[:8]}"
    
    config_dict = {
        "underlying": underlying,
        "start": target_date,
        "end": target_date,
        "data_source": "real_scraper",
        "decision_interval_minutes": args.decision_interval,
        "target_delta": args.target_delta,
        "target_dte": args.target_dte,
    }
    
    print(f"\nCreating backtest run: {run_id}")
    
    with get_db_session() as session:
        db_run = create_backtest_run(
            session=session,
            run_id=run_id,
            underlying=underlying,
            start_ts=start_ts,
            end_ts=end_ts,
            data_source="real_scraper",
            decision_interval_minutes=args.decision_interval,
            config_json=config_dict,
        )
        session.commit()
    
    config = CallSimulationConfig(
        underlying=underlying,
        start=start_ts,
        end=end_ts,
        timeframe="1h",
        decision_interval_bars=max(1, args.decision_interval // 60),
        target_delta=args.target_delta,
        target_dte=args.target_dte,
    )
    
    ds = RealScraperDataSource(underlying=underlying, start_ts=start_ts, end_ts=end_ts)
    simulator = CoveredCallSimulator(data_source=ds, config=config)
    
    try:
        print("\nRunning simulation...")
        result = simulator.simulate_policy(policy=always_trade_policy, size=1.0)
        
        trades = result.trades if hasattr(result, "trades") else []
        total_pnl = sum(t.pnl for t in trades) if trades else 0.0
        num_trades = len(trades)
        
        print(f"\nSimulation complete:")
        print(f"  Trades executed: {num_trades}")
        print(f"  Total PnL:       {total_pnl:.4f}")
        
        metrics_by_style = {
            "default": {
                "initial_equity": None,
                "final_equity": None,
                "net_profit_usd": total_pnl if trades else None,
                "net_profit_pct": 0.0,
                "max_drawdown_pct": 0.0,
                "num_trades": num_trades,
                "win_rate": 0.0,
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
            }
        }
        
        with get_db_session() as session:
            complete_run(
                session=session,
                run_id=run_id,
                metrics_by_style=metrics_by_style,
                chains_by_style={},
                primary_exit_style="default",
            )
            session.commit()
        
        print(f"\n" + "=" * 60)
        print("BACKTEST RESULTS")
        print("=" * 60)
        print(f"Run ID:            {run_id}")
        print(f"Status:            finished")
        print(f"Data source:       real_scraper")
        print(f"\nMetrics (default exit style):")
        for key, value in metrics_by_style["default"].items():
            print(f"  {key}: {value}")
        print("=" * 60)
        
        print(f"\nVerify in database:")
        print(f"  psql $DATABASE_URL -c \"SELECT run_id, status, data_source FROM backtest_runs WHERE run_id = '{run_id}';\"")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        
        with get_db_session() as session:
            fail_run(session, run_id, str(e))
            session.commit()
        
        print(f"\nBacktest failed: {e}")
        sys.exit(1)
    finally:
        ds.close()


if __name__ == "__main__":
    main()
