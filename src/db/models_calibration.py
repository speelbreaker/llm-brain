"""
SQLAlchemy model for calibration history persistence.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    BigInteger,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Index,
)
from sqlalchemy.sql import func

from src.db import Base, get_db_session


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
            "num_samples": self.num_samples,
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
            num_samples=entry.num_samples,
        )
        db.add(row)
        db.flush()
        return row.id


def get_latest_calibration(
    underlying: str,
    dte_min: int,
    dte_max: int,
) -> Optional[CalibrationHistoryEntry]:
    """
    Get the most recent calibration for the given underlying and DTE range.
    
    Returns:
        CalibrationHistoryEntry or None if not found.
    """
    with get_db_session() as db:
        row = (
            db.query(CalibrationHistory)
            .filter(
                CalibrationHistory.underlying == underlying.upper(),
                CalibrationHistory.dte_min == dte_min,
                CalibrationHistory.dte_max == dte_max,
            )
            .order_by(CalibrationHistory.created_at.desc())
            .first()
        )
        
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
            num_samples=row.num_samples,
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
                num_samples=row.num_samples,
                created_at=row.created_at,
            )
            for row in rows
        ]
