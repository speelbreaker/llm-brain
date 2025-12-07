"""
SQLAlchemy models for backtest persistence.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from src.db import Base


class BacktestRun(Base):
    """
    Represents a single backtest run.
    One row per backtest execution.
    """
    __tablename__ = "backtest_runs"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    run_id = Column(String(128), unique=True, nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    
    status = Column(String(32), nullable=False, default="queued")
    
    underlying = Column(String(32), nullable=False)
    data_source = Column(String(64), nullable=False, default="synthetic")
    start_ts = Column(DateTime(timezone=True), nullable=False)
    end_ts = Column(DateTime(timezone=True), nullable=False)
    decision_interval_minutes = Column(Integer, nullable=True)
    primary_exit_style = Column(String(64), nullable=True)
    
    initial_equity = Column(Float, nullable=True)
    final_equity_primary = Column(Float, nullable=True)
    net_profit_pct_primary = Column(Float, nullable=True)
    max_drawdown_pct_primary = Column(Float, nullable=True)
    sharpe_primary = Column(Float, nullable=True)
    sortino_primary = Column(Float, nullable=True)
    
    config_json = Column(JSONB, nullable=True)
    notes = Column(Text, nullable=True)
    
    metrics = relationship("BacktestMetric", back_populates="run", cascade="all, delete-orphan")
    chains = relationship("BacktestChain", back_populates="run", cascade="all, delete-orphan")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "id": self.id,
            "run_id": self.run_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "status": self.status,
            "underlying": self.underlying,
            "data_source": self.data_source,
            "start_ts": self.start_ts.isoformat() if self.start_ts else None,
            "end_ts": self.end_ts.isoformat() if self.end_ts else None,
            "decision_interval_minutes": self.decision_interval_minutes,
            "primary_exit_style": self.primary_exit_style,
            "initial_equity": self.initial_equity,
            "final_equity_primary": self.final_equity_primary,
            "net_profit_pct_primary": self.net_profit_pct_primary,
            "max_drawdown_pct_primary": self.max_drawdown_pct_primary,
            "sharpe_primary": self.sharpe_primary,
            "sortino_primary": self.sortino_primary,
            "config_json": self.config_json,
            "notes": self.notes,
        }


class BacktestMetric(Base):
    """
    Per-run, per-exit-style metrics.
    Each run can have metrics for hold_to_expiry, tp_and_roll, etc.
    """
    __tablename__ = "backtest_metrics"
    __table_args__ = (
        UniqueConstraint("run_id", "exit_style", name="uq_backtest_metrics_run_exit"),
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    run_id = Column(BigInteger, ForeignKey("backtest_runs.id", ondelete="CASCADE"), nullable=False, index=True)
    exit_style = Column(String(64), nullable=False)
    is_primary = Column(Boolean, default=False, nullable=False)
    
    initial_equity = Column(Float, nullable=True)
    final_equity = Column(Float, nullable=True)
    net_profit_usd = Column(Float, nullable=True)
    net_profit_pct = Column(Float, nullable=True)
    hodl_profit_usd = Column(Float, nullable=True)
    hodl_profit_pct = Column(Float, nullable=True)
    max_drawdown_pct = Column(Float, nullable=True)
    max_drawdown_usd = Column(Float, nullable=True)
    num_trades = Column(Integer, nullable=True)
    win_rate = Column(Float, nullable=True)
    avg_trade_usd = Column(Float, nullable=True)
    profit_factor = Column(Float, nullable=True)
    gross_profit = Column(Float, nullable=True)
    gross_loss = Column(Float, nullable=True)
    avg_winner = Column(Float, nullable=True)
    avg_loser = Column(Float, nullable=True)
    sharpe_ratio = Column(Float, nullable=True)
    sortino_ratio = Column(Float, nullable=True)
    final_pnl = Column(Float, nullable=True)
    final_pnl_vs_hodl = Column(Float, nullable=True)
    avg_pnl = Column(Float, nullable=True)
    
    run = relationship("BacktestRun", back_populates="metrics")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "id": self.id,
            "exit_style": self.exit_style,
            "is_primary": self.is_primary,
            "initial_equity": self.initial_equity,
            "final_equity": self.final_equity,
            "net_profit_usd": self.net_profit_usd,
            "net_profit_pct": self.net_profit_pct,
            "hodl_profit_usd": self.hodl_profit_usd,
            "hodl_profit_pct": self.hodl_profit_pct,
            "max_drawdown_pct": self.max_drawdown_pct,
            "max_drawdown_usd": self.max_drawdown_usd,
            "num_trades": self.num_trades,
            "win_rate": self.win_rate,
            "avg_trade_usd": self.avg_trade_usd,
            "profit_factor": self.profit_factor,
            "gross_profit": self.gross_profit,
            "gross_loss": self.gross_loss,
            "avg_winner": self.avg_winner,
            "avg_loser": self.avg_loser,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "final_pnl": self.final_pnl,
            "final_pnl_vs_hodl": self.final_pnl_vs_hodl,
            "avg_pnl": self.avg_pnl,
        }


class BacktestChain(Base):
    """
    Multi-leg chain summaries for a backtest run.
    Stores individual trade chains with their metrics.
    """
    __tablename__ = "backtest_chains"
    __table_args__ = (
        Index("ix_backtest_chains_run_id", "run_id"),
        Index("ix_backtest_chains_run_exit", "run_id", "exit_style"),
        Index("ix_backtest_chains_run_time", "run_id", "decision_time"),
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    run_id = Column(BigInteger, ForeignKey("backtest_runs.id", ondelete="CASCADE"), nullable=False)
    exit_style = Column(String(64), nullable=False)
    decision_time = Column(DateTime(timezone=True), nullable=True)
    underlying = Column(String(32), nullable=True)
    
    chain_label = Column(String(256), nullable=True)
    num_legs = Column(Integer, nullable=True)
    num_rolls = Column(Integer, nullable=True)
    
    total_pnl_usd = Column(Float, nullable=True)
    pnl_vs_hodl_usd = Column(Float, nullable=True)
    max_drawdown_pct = Column(Float, nullable=True)
    max_drawdown_usd = Column(Float, nullable=True)
    
    details_json = Column(JSONB, nullable=True)
    
    run = relationship("BacktestRun", back_populates="chains")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "id": self.id,
            "exit_style": self.exit_style,
            "decision_time": self.decision_time.isoformat() if self.decision_time else None,
            "underlying": self.underlying,
            "chain_label": self.chain_label,
            "num_legs": self.num_legs,
            "num_rolls": self.num_rolls,
            "total_pnl_usd": self.total_pnl_usd,
            "pnl_vs_hodl_usd": self.pnl_vs_hodl_usd,
            "max_drawdown_pct": self.max_drawdown_pct,
            "max_drawdown_usd": self.max_drawdown_usd,
            "details_json": self.details_json,
        }
