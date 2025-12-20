from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class OptionQuote:
    instrument_name: str
    option_type: str  # "call" | "put"
    strike: float
    expiry_ts: int
    mark_price: float
    mark_iv: Optional[float] = None
    delta: Optional[float] = None
    bid: Optional[float] = None
    ask: Optional[float] = None


@dataclass(frozen=True)
class MarketSnapshot:
    ts: int
    underlying: str
    spot: Optional[float] = None
    options: List[OptionQuote] = field(default_factory=list)


@dataclass(frozen=True)
class FillModelConfig:
    slippage_bps: float = 0.0
    use_mid: bool = True


@dataclass(frozen=True)
class Trade:
    open_ts: int
    close_ts: int
    pnl: float
    pnl_pct: float
    is_valid: bool = True
    data_quality_status: str = "ok"  # ok|missing_close_quote|missing_open_quote|stale_quote|expired_no_quote
    metadata: Dict[str, Any] = field(default_factory=dict)


TradeResult = Trade
