"""
Generic types for bot strategy evaluations.
"""
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class StrategyCriterion(BaseModel):
    """A single metric check for a strategy."""
    metric: str = Field(..., description="Metric name (e.g., vrp_30d, rsi_14d)")
    value: Optional[float] = Field(None, description="Current numeric value, if available")
    min: Optional[float] = Field(None, description="Minimum threshold")
    max: Optional[float] = Field(None, description="Maximum threshold")
    ok: bool = Field(..., description="True if condition passed, False if failed or missing")
    note: Optional[str] = Field(None, description="Short explanation: missing_data, below_min, etc.")


class StrategyEvaluation(BaseModel):
    """Result of evaluating a strategy against current market conditions."""
    bot_name: str = Field(..., description="Bot display name (e.g., GregBot)")
    expert_id: str = Field(..., description="Expert identifier (e.g., greg_mandolini)")
    underlying: str = Field(..., description="Underlying asset (BTC, ETH)")
    strategy_key: str = Field(..., description="Stable identifier (e.g., atm_straddle_30d)")
    label: str = Field(..., description="Human-friendly name (e.g., 30d ATM Straddle)")
    status: Literal["pass", "blocked", "no_data"] = Field(
        ..., 
        description="pass=meets all criteria, blocked=fails criteria, no_data=missing data"
    )
    summary: str = Field(..., description="1-2 sentence explanation of why it passes or doesn't")
    criteria: List[StrategyCriterion] = Field(
        default_factory=list, 
        description="List of metric checks"
    )
    debug: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Optional debug info (rule path, sensor snapshot, etc.)"
    )
