"""
Database-backed backtest service.
Handles creating, updating, and querying backtest runs in PostgreSQL.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from src.db import get_db_session
from src.db.models_backtest import BacktestRun, BacktestMetric, BacktestChain


def generate_run_id(underlying: str) -> str:
    """Generate a unique run ID."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    short_uuid = uuid.uuid4().hex[:8]
    return f"{ts}_{underlying}_{short_uuid}"


def create_backtest_run(
    db: Session,
    underlying: str,
    start_ts: datetime,
    end_ts: datetime,
    data_source: str = "synthetic",
    decision_interval_minutes: Optional[int] = None,
    primary_exit_style: Optional[str] = None,
    config_json: Optional[Dict[str, Any]] = None,
) -> BacktestRun:
    """Create a new backtest run in queued status."""
    run_id = generate_run_id(underlying)
    
    run = BacktestRun(
        run_id=run_id,
        status="queued",
        underlying=underlying,
        data_source=data_source,
        start_ts=start_ts,
        end_ts=end_ts,
        decision_interval_minutes=decision_interval_minutes,
        primary_exit_style=primary_exit_style,
        config_json=config_json,
    )
    
    db.add(run)
    db.commit()
    db.refresh(run)
    return run


def update_run_status(
    db: Session,
    run_id: str,
    status: str,
    error: Optional[str] = None,
) -> Optional[BacktestRun]:
    """Update the status of a backtest run."""
    run = db.query(BacktestRun).filter(BacktestRun.run_id == run_id).first()
    if run:
        run.status = status
        if error:
            run.notes = error
        db.commit()
        db.refresh(run)
    return run


def save_run_metrics(
    db: Session,
    run: BacktestRun,
    metrics_by_style: Dict[str, Dict[str, Any]],
    primary_exit_style: str = "tp_and_roll",
) -> None:
    """Save metrics for a backtest run (one row per exit style)."""
    for exit_style, metrics in metrics_by_style.items():
        is_primary = (exit_style == primary_exit_style)
        
        metric = BacktestMetric(
            run_id=run.id,
            exit_style=exit_style,
            is_primary=is_primary,
            initial_equity=metrics.get("initial_equity"),
            final_equity=metrics.get("final_equity"),
            net_profit_usd=metrics.get("net_profit_usd"),
            net_profit_pct=metrics.get("net_profit_pct"),
            hodl_profit_usd=metrics.get("hodl_profit_usd"),
            hodl_profit_pct=metrics.get("hodl_profit_pct"),
            max_drawdown_pct=metrics.get("max_drawdown_pct"),
            max_drawdown_usd=metrics.get("max_drawdown_usd"),
            num_trades=metrics.get("num_trades"),
            win_rate=metrics.get("win_rate"),
            avg_trade_usd=metrics.get("avg_trade_usd"),
            profit_factor=metrics.get("profit_factor"),
            gross_profit=metrics.get("gross_profit"),
            gross_loss=metrics.get("gross_loss"),
            avg_winner=metrics.get("avg_winner"),
            avg_loser=metrics.get("avg_loser"),
            sharpe_ratio=metrics.get("sharpe_ratio"),
            sortino_ratio=metrics.get("sortino_ratio"),
            final_pnl=metrics.get("final_pnl"),
            final_pnl_vs_hodl=metrics.get("final_pnl_vs_hodl"),
            avg_pnl=metrics.get("avg_pnl"),
        )
        db.add(metric)
        
        if is_primary:
            run.primary_exit_style = exit_style
            run.initial_equity = metrics.get("initial_equity")
            run.final_equity_primary = metrics.get("final_equity")
            run.net_profit_pct_primary = metrics.get("net_profit_pct")
            run.max_drawdown_pct_primary = metrics.get("max_drawdown_pct")
            run.sharpe_primary = metrics.get("sharpe_ratio")
            run.sortino_primary = metrics.get("sortino_ratio")
    
    db.commit()


def save_run_chains(
    db: Session,
    run: BacktestRun,
    chains_by_style: Dict[str, List[Dict[str, Any]]],
    underlying: str,
) -> None:
    """Save chain records for a backtest run."""
    for exit_style, chains in chains_by_style.items():
        for chain_data in chains:
            decision_time = None
            if chain_data.get("open_time"):
                try:
                    decision_time = datetime.fromisoformat(chain_data["open_time"].replace("Z", "+00:00"))
                except (ValueError, TypeError):
                    pass
            
            chain = BacktestChain(
                run_id=run.id,
                exit_style=exit_style,
                decision_time=decision_time,
                underlying=underlying,
                chain_label=chain_data.get("instrument_name"),
                num_legs=chain_data.get("num_legs", 1),
                num_rolls=chain_data.get("num_rolls", 0),
                total_pnl_usd=chain_data.get("pnl"),
                pnl_vs_hodl_usd=chain_data.get("pnl_vs_hodl"),
                max_drawdown_pct=chain_data.get("max_drawdown_pct"),
                max_drawdown_usd=chain_data.get("max_drawdown_usd"),
                details_json=chain_data,
            )
            db.add(chain)
    
    db.commit()


def complete_run(
    db: Session,
    run: BacktestRun,
    metrics_by_style: Dict[str, Dict[str, Any]],
    chains_by_style: Dict[str, List[Dict[str, Any]]],
    primary_exit_style: str = "tp_and_roll",
) -> BacktestRun:
    """Mark a run as finished and save all metrics and chains."""
    save_run_metrics(db, run, metrics_by_style, primary_exit_style)
    save_run_chains(db, run, chains_by_style, run.underlying)
    
    run.status = "finished"
    db.commit()
    db.refresh(run)
    return run


def fail_run(db: Session, run: BacktestRun, error: str) -> BacktestRun:
    """Mark a run as failed with an error message."""
    run.status = "failed"
    run.notes = error
    db.commit()
    db.refresh(run)
    return run


def list_runs(
    db: Session,
    underlying: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 100,
) -> List[BacktestRun]:
    """List backtest runs with optional filters."""
    query = db.query(BacktestRun).order_by(BacktestRun.created_at.desc())
    
    if underlying:
        query = query.filter(BacktestRun.underlying == underlying)
    if status:
        query = query.filter(BacktestRun.status == status)
    
    return query.limit(limit).all()


def get_run_by_id(db: Session, run_id: str) -> Optional[BacktestRun]:
    """Get a backtest run by its run_id string."""
    return db.query(BacktestRun).filter(BacktestRun.run_id == run_id).first()


def get_run_with_details(db: Session, run_id: str) -> Optional[Dict[str, Any]]:
    """Get a run with all its metrics and chains."""
    run = db.query(BacktestRun).filter(BacktestRun.run_id == run_id).first()
    if not run:
        return None
    
    metrics_dict = {}
    for metric in run.metrics:
        metrics_dict[metric.exit_style] = metric.to_dict()
    
    chains_dict: Dict[str, List[Dict[str, Any]]] = {}
    for chain in sorted(run.chains, key=lambda c: c.decision_time or datetime.min, reverse=True)[:50]:
        if chain.exit_style not in chains_dict:
            chains_dict[chain.exit_style] = []
        chains_dict[chain.exit_style].append(chain.to_dict())
    
    return {
        "run": run.to_dict(),
        "metrics": metrics_dict,
        "chains": chains_dict,
    }


def delete_run(db: Session, run_id: str) -> bool:
    """Delete a backtest run and all its associated data."""
    run = db.query(BacktestRun).filter(BacktestRun.run_id == run_id).first()
    if run:
        db.delete(run)
        db.commit()
        return True
    return False
