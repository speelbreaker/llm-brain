"""
SQLAlchemy model for Greg decision logging.

Tracks both suggestions (advisory decisions) and executions.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    Index,
)
from sqlalchemy.sql import func

from src.db import Base, get_db_session


class GregActionType(str, Enum):
    """Types of Greg decision actions."""
    OPEN = "OPEN"
    HEDGE = "HEDGE"
    TAKE_PROFIT = "TAKE_PROFIT"
    ASSIGN = "ASSIGN"
    ROLL = "ROLL"
    CLOSE = "CLOSE"
    HOLD = "HOLD"
    IGNORE = "IGNORE"


class GregDecisionLog(Base):
    """
    Stores Greg strategy decisions for tracking and analysis.
    
    Used to log:
    - Suggestions: When the Position Management engine computes a recommendation
    - Executions: When a user actually executes the suggested action
    - Ignores: When a user explicitly dismisses a suggestion
    """
    __tablename__ = "greg_decision_log"
    __table_args__ = (
        Index("ix_greg_decision_log_underlying", "underlying"),
        Index("ix_greg_decision_log_strategy", "strategy_type"),
        Index("ix_greg_decision_log_timestamp", "timestamp"),
        Index("ix_greg_decision_log_position", "position_id"),
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime(timezone=True), default=func.now(), nullable=False)
    
    underlying = Column(String(10), nullable=False)
    strategy_type = Column(String(50), nullable=False)
    position_id = Column(String(100), nullable=False)
    
    action_type = Column(String(20), nullable=False)
    mode = Column(String(20), nullable=False)
    
    suggested = Column(Boolean, default=True, nullable=False)
    executed = Column(Boolean, default=False, nullable=False)
    
    reason = Column(Text, nullable=True)
    
    vrp_30d = Column(Float, nullable=True)
    chop_factor_7d = Column(Float, nullable=True)
    adx_14d = Column(Float, nullable=True)
    term_structure_spread = Column(Float, nullable=True)
    skew_25d = Column(Float, nullable=True)
    rsi_14d = Column(Float, nullable=True)
    
    pnl_pct = Column(Float, nullable=True)
    pnl_usd = Column(Float, nullable=True)
    net_delta = Column(Float, nullable=True)
    spot_price = Column(Float, nullable=True)
    
    order_ids = Column(Text, nullable=True)
    extra_info = Column(Text, nullable=True)


def log_greg_decision(
    underlying: str,
    strategy_type: str,
    position_id: str,
    action_type: str,
    mode: str,
    suggested: bool = True,
    executed: bool = False,
    reason: Optional[str] = None,
    vrp_30d: Optional[float] = None,
    chop_factor_7d: Optional[float] = None,
    adx_14d: Optional[float] = None,
    term_structure_spread: Optional[float] = None,
    skew_25d: Optional[float] = None,
    rsi_14d: Optional[float] = None,
    pnl_pct: Optional[float] = None,
    pnl_usd: Optional[float] = None,
    net_delta: Optional[float] = None,
    spot_price: Optional[float] = None,
    order_ids: Optional[str] = None,
    extra_info: Optional[str] = None,
) -> Optional[GregDecisionLog]:
    """
    Log a Greg decision (suggestion or execution).
    
    Returns the created log entry, or None if database not available.
    """
    try:
        with get_db_session() as session:
            entry = GregDecisionLog(
                timestamp=datetime.now(timezone.utc),
                underlying=underlying,
                strategy_type=strategy_type,
                position_id=position_id,
                action_type=action_type,
                mode=mode,
                suggested=suggested,
                executed=executed,
                reason=reason,
                vrp_30d=vrp_30d,
                chop_factor_7d=chop_factor_7d,
                adx_14d=adx_14d,
                term_structure_spread=term_structure_spread,
                skew_25d=skew_25d,
                rsi_14d=rsi_14d,
                pnl_pct=pnl_pct,
                pnl_usd=pnl_usd,
                net_delta=net_delta,
                spot_price=spot_price,
                order_ids=order_ids,
                extra_info=extra_info,
            )
            session.add(entry)
            session.commit()
            session.refresh(entry)
            return entry
    except Exception as e:
        print(f"[GregDecisionLog] Failed to log decision: {e}")
        return None


def mark_decision_executed(
    position_id: str,
    action_type: str,
    order_ids: Optional[str] = None,
    extra_info: Optional[str] = None,
) -> bool:
    """
    Mark a previously logged suggestion as executed.
    
    Finds the most recent suggestion for this position/action and updates it.
    Returns True if updated, False otherwise.
    """
    try:
        with get_db_session() as session:
            entry = (
                session.query(GregDecisionLog)
                .filter(
                    GregDecisionLog.position_id == position_id,
                    GregDecisionLog.action_type == action_type,
                    GregDecisionLog.suggested == True,
                    GregDecisionLog.executed == False,
                )
                .order_by(GregDecisionLog.timestamp.desc())
                .first()
            )
            
            if entry:
                entry.executed = True
                if order_ids:
                    entry.order_ids = order_ids
                if extra_info:
                    entry.extra_info = extra_info
                session.commit()
                return True
            return False
    except Exception as e:
        print(f"[GregDecisionLog] Failed to mark executed: {e}")
        return False


def get_decision_history(
    underlying: Optional[str] = None,
    strategy_type: Optional[str] = None,
    from_date: Optional[datetime] = None,
    to_date: Optional[datetime] = None,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    """
    Query decision log history with filters.
    
    Returns list of dicts with decision data.
    """
    try:
        with get_db_session() as session:
            query = session.query(GregDecisionLog)
            
            if underlying:
                query = query.filter(GregDecisionLog.underlying == underlying)
            if strategy_type:
                query = query.filter(GregDecisionLog.strategy_type == strategy_type)
            if from_date:
                query = query.filter(GregDecisionLog.timestamp >= from_date)
            if to_date:
                query = query.filter(GregDecisionLog.timestamp <= to_date)
            
            query = query.order_by(GregDecisionLog.timestamp.desc()).limit(limit)
            entries = query.all()
            
            return [
                {
                    "id": e.id,
                    "timestamp": e.timestamp.isoformat() if e.timestamp else None,
                    "underlying": e.underlying,
                    "strategy_type": e.strategy_type,
                    "position_id": e.position_id,
                    "action_type": e.action_type,
                    "mode": e.mode,
                    "suggested": e.suggested,
                    "executed": e.executed,
                    "reason": e.reason,
                    "vrp_30d": e.vrp_30d,
                    "pnl_pct": e.pnl_pct,
                    "pnl_usd": e.pnl_usd,
                    "net_delta": e.net_delta,
                    "order_ids": e.order_ids,
                }
                for e in entries
            ]
    except Exception as e:
        print(f"[GregDecisionLog] Failed to get history: {e}")
        return []


def get_decision_stats(
    underlying: Optional[str] = None,
    from_date: Optional[datetime] = None,
    to_date: Optional[datetime] = None,
) -> Dict[str, Any]:
    """
    Get summary statistics for decisions.
    
    Returns counts grouped by strategy and action type.
    """
    try:
        with get_db_session() as session:
            query = session.query(GregDecisionLog)
            
            if underlying:
                query = query.filter(GregDecisionLog.underlying == underlying)
            if from_date:
                query = query.filter(GregDecisionLog.timestamp >= from_date)
            if to_date:
                query = query.filter(GregDecisionLog.timestamp <= to_date)
            
            entries = query.all()
            
            stats: Dict[str, Any] = {
                "total_suggestions": 0,
                "total_executed": 0,
                "by_strategy": {},
                "by_action": {},
            }
            
            for e in entries:
                if e.suggested:
                    stats["total_suggestions"] += 1
                if e.executed:
                    stats["total_executed"] += 1
                
                if e.strategy_type not in stats["by_strategy"]:
                    stats["by_strategy"][e.strategy_type] = {
                        "suggestions": 0, "executions": 0, "avg_pnl_pct": []
                    }
                if e.suggested:
                    stats["by_strategy"][e.strategy_type]["suggestions"] += 1
                if e.executed:
                    stats["by_strategy"][e.strategy_type]["executions"] += 1
                if e.pnl_pct is not None:
                    stats["by_strategy"][e.strategy_type]["avg_pnl_pct"].append(e.pnl_pct)
                
                if e.action_type not in stats["by_action"]:
                    stats["by_action"][e.action_type] = {"suggestions": 0, "executions": 0}
                if e.suggested:
                    stats["by_action"][e.action_type]["suggestions"] += 1
                if e.executed:
                    stats["by_action"][e.action_type]["executions"] += 1
            
            for strat, data in stats["by_strategy"].items():
                pnl_list = data["avg_pnl_pct"]
                data["avg_pnl_pct"] = sum(pnl_list) / len(pnl_list) if pnl_list else None
            
            return stats
    except Exception as e:
        print(f"[GregDecisionLog] Failed to get stats: {e}")
        return {"error": str(e)}
