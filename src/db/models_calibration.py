"""
SQLAlchemy model for calibration history persistence.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Tuple

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


MIN_REASONABLE_MULT = 0.7
MAX_REASONABLE_MULT = 1.6
MAX_VEGA_WEIGHTED_MAE = 200.0
MAX_UNWEIGHTED_MAE = 400.0


def assess_calibration_realism(
    multiplier: float,
    mae_pct: Optional[float],
    vega_weighted_mae_pct: Optional[float],
    data_quality_status: str = "ok",
) -> Tuple[str, str]:
    """
    Assess whether a calibration result is realistic.
    
    Returns:
        (status, reason) where status is 'ok', 'degraded', or 'failed'.
        
    Status levels:
    - 'ok': Calibration is within thresholds and data quality is good
    - 'degraded': Calibration is within thresholds but data quality has issues
    - 'failed': Calibration is unrealistic (multiplier at boundary, MAE too high) 
                or data quality failed completely
    """
    if data_quality_status == "failed":
        return ("failed", "Data quality failed (schema/coverage issues)")
    
    issues = []
    
    if multiplier <= MIN_REASONABLE_MULT:
        issues.append(f"multiplier at lower boundary ({multiplier:.4f})")
    elif multiplier >= MAX_REASONABLE_MULT:
        issues.append(f"multiplier at upper boundary ({multiplier:.4f})")
    
    if vega_weighted_mae_pct is not None and vega_weighted_mae_pct > MAX_VEGA_WEIGHTED_MAE:
        issues.append(f"vMAE too high ({vega_weighted_mae_pct:.1f}%)")
    
    if mae_pct is not None and mae_pct > MAX_UNWEIGHTED_MAE:
        issues.append(f"MAE too high ({mae_pct:.1f}%)")
    
    if issues:
        reason = "Unrealistic auto-calibration: " + ", ".join(issues)
        return ("failed", reason)
    
    if data_quality_status == "degraded":
        return ("degraded", "Data quality degraded; use with caution")
    
    return ("ok", "Calibration within thresholds")


class CalibrationHistory(Base):
    """
    Stores auto-calculated IV multipliers over time.
    Keyed by underlying and DTE range.
    """
    __tablename__ = "calibration_history"
    __table_args__ = (
        Index("ix_calibration_history_underlying", "underlying"),
        Index("ix_calibration_history_lookup", "underlying", "dte_min", "dte_max"),
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    underlying = Column(String(16), nullable=False)
    dte_min = Column(Integer, nullable=False)
    dte_max = Column(Integer, nullable=False)
    lookback_days = Column(Integer, nullable=False)
    multiplier = Column(Float, nullable=False)
    mae_pct = Column(Float, nullable=False)
    num_samples = Column(Integer, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    vega_weighted_mae_pct = Column(Float, nullable=True)
    bias_pct = Column(Float, nullable=True)
    source = Column(String(32), nullable=True, default="harvested")
    status = Column(String(16), nullable=True, default="ok")
    reason = Column(Text, nullable=True)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "id": self.id,
            "underlying": self.underlying,
            "dte_min": self.dte_min,
            "dte_max": self.dte_max,
            "lookback_days": self.lookback_days,
            "multiplier": self.multiplier,
            "mae_pct": self.mae_pct,
            "vega_weighted_mae_pct": self.vega_weighted_mae_pct,
            "bias_pct": self.bias_pct,
            "num_samples": self.num_samples,
            "source": self.source,
            "status": self.status,
            "reason": self.reason,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


@dataclass
class CalibrationHistoryEntry:
    """Data class for calibration history entries."""
    underlying: str
    dte_min: int
    dte_max: int
    lookback_days: int
    multiplier: float
    mae_pct: float
    num_samples: int
    id: Optional[int] = None
    created_at: Optional[datetime] = None
    vega_weighted_mae_pct: Optional[float] = None
    bias_pct: Optional[float] = None
    source: str = "harvested"
    status: str = "ok"
    reason: Optional[str] = None


def insert_calibration_history(entry: CalibrationHistoryEntry) -> int:
    """
    Insert a new calibration history entry.
    
    Returns:
        The ID of the inserted row.
    """
    with get_db_session() as db:
        row = CalibrationHistory(
            underlying=entry.underlying.upper(),
            dte_min=entry.dte_min,
            dte_max=entry.dte_max,
            lookback_days=entry.lookback_days,
            multiplier=entry.multiplier,
            mae_pct=entry.mae_pct,
            vega_weighted_mae_pct=entry.vega_weighted_mae_pct,
            bias_pct=entry.bias_pct,
            num_samples=entry.num_samples,
            source=entry.source,
            status=entry.status,
            reason=entry.reason,
        )
        db.add(row)
        db.flush()
        return row.id


def get_latest_calibration(
    underlying: str,
    dte_min: int,
    dte_max: int,
    skip_failed: bool = False,
) -> Optional[CalibrationHistoryEntry]:
    """
    Get the most recent calibration for the given underlying and DTE range.
    
    Args:
        underlying: BTC or ETH
        dte_min: Minimum DTE for calibration range
        dte_max: Maximum DTE for calibration range
        skip_failed: If True, skip entries with status='failed' and return the latest OK/degraded entry
    
    Returns:
        CalibrationHistoryEntry or None if not found.
    """
    with get_db_session() as db:
        query = (
            db.query(CalibrationHistory)
            .filter(
                CalibrationHistory.underlying == underlying.upper(),
                CalibrationHistory.dte_min == dte_min,
                CalibrationHistory.dte_max == dte_max,
            )
        )
        
        if skip_failed:
            query = query.filter(CalibrationHistory.status != "failed")
        
        row = query.order_by(CalibrationHistory.created_at.desc()).first()
        
        if row is None:
            return None
        
        return CalibrationHistoryEntry(
            id=row.id,
            underlying=row.underlying,
            dte_min=row.dte_min,
            dte_max=row.dte_max,
            lookback_days=row.lookback_days,
            multiplier=row.multiplier,
            mae_pct=row.mae_pct,
            vega_weighted_mae_pct=row.vega_weighted_mae_pct,
            bias_pct=row.bias_pct,
            num_samples=row.num_samples,
            source=row.source,
            status=row.status,
            reason=row.reason,
            created_at=row.created_at,
        )


def list_recent_calibrations(
    underlying: str,
    limit: int = 20,
) -> List[CalibrationHistoryEntry]:
    """
    List recent calibration entries for the given underlying.
    
    Returns:
        List of CalibrationHistoryEntry ordered by created_at descending.
    """
    with get_db_session() as db:
        rows = (
            db.query(CalibrationHistory)
            .filter(CalibrationHistory.underlying == underlying.upper())
            .order_by(CalibrationHistory.created_at.desc())
            .limit(limit)
            .all()
        )
        
        return [
            CalibrationHistoryEntry(
                id=row.id,
                underlying=row.underlying,
                dte_min=row.dte_min,
                dte_max=row.dte_max,
                lookback_days=row.lookback_days,
                multiplier=row.multiplier,
                mae_pct=row.mae_pct,
                vega_weighted_mae_pct=row.vega_weighted_mae_pct,
                bias_pct=row.bias_pct,
                num_samples=row.num_samples,
                source=row.source,
                status=row.status,
                reason=row.reason,
                created_at=row.created_at,
            )
            for row in rows
        ]
