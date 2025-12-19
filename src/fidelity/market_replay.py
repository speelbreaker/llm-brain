from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Protocol, Literal, Dict, Any

from src.backtest.types import OptionSnapshot


FillSide = Literal["buy", "sell"]


@dataclass(frozen=True)
class FillModelConfig:
    """Minimal fill model configuration (Phase 1).

    Phase 1: Fill at mark price +/- fixed bps.
    """

    bps: float = 0.0


class FillModel(Protocol):
    def fill_price(self, *, mark_price: float, side: FillSide) -> float: ...


class FixedBpsFillModel:
    def __init__(self, cfg: FillModelConfig):
        self._bps = float(cfg.bps)

    def fill_price(self, *, mark_price: float, side: FillSide) -> float:
        if mark_price <= 0:
            return 0.0
        adj = self._bps / 10_000.0
        if side == "buy":
            return mark_price * (1.0 + adj)
        return mark_price * (1.0 - adj)


@dataclass(frozen=True)
class MarketSnapshot:
    time: datetime
    underlying: str
    spot: float
    option_chain: List[OptionSnapshot]


class MarketReplay(Protocol):
    """A common interface for replaying markets at a timestamp.

    Both LiveReplayMarket and SyntheticReplayMarket must implement this.
    """

    underlying: str

    def snapshot(self, t: datetime) -> MarketSnapshot: ...

    def fill_model(self) -> FillModel: ...

    def meta(self) -> Dict[str, Any]: ...


class LiveReplayMarket:
    """Replays harvested Deribit snapshots.

    P0 implementation uses DeribitDataSource-compatible interface.
    """

    def __init__(
        self,
        *,
        underlying: str,
        ds: Any,
        fill_cfg: Optional[FillModelConfig] = None,
        settlement_ccy: str = "USDC",
        margin_type: str = "linear",
    ):
        self.underlying = underlying
        self._ds = ds
        self._fill = FixedBpsFillModel(fill_cfg or FillModelConfig())
        self._settlement_ccy = settlement_ccy
        self._margin_type = margin_type

    def snapshot(self, t: datetime) -> MarketSnapshot:
        spot_df = self._ds.get_spot_ohlc(
            underlying=self.underlying,
            start=t,
            end=t,
            timeframe="1h",
        )
        spot = float(spot_df["close"].iloc[-1]) if not spot_df.empty else 0.0

        chain = self._ds.list_option_chain(
            underlying=self.underlying,
            as_of=t,
            settlement_ccy=self._settlement_ccy,
            margin_type=self._margin_type,
        )
        return MarketSnapshot(time=t, underlying=self.underlying, spot=spot, option_chain=chain)

    def fill_model(self) -> FillModel:
        return self._fill

    def meta(self) -> Dict[str, Any]:
        return {
            "type": "live_replay",
            "underlying": self.underlying,
            "ds_class": self._ds.__class__.__name__,
            "settlement_ccy": self._settlement_ccy,
            "margin_type": self._margin_type,
        }


class SyntheticReplayMarket:
    """Replays synthetic universe snapshots aligned to the same timestamps.

    P0 implementation uses backtest.state_builder synthetic grid generation.
    """

    def __init__(
        self,
        *,
        underlying: str,
        ds: Any,
        cfg: Any,
        fill_cfg: Optional[FillModelConfig] = None,
    ):
        self.underlying = underlying
        self._ds = ds
        self._cfg = cfg
        self._fill = FixedBpsFillModel(fill_cfg or FillModelConfig())

    def snapshot(self, t: datetime) -> MarketSnapshot:
        from src.backtest.state_builder import build_historical_state

        state = build_historical_state(self._ds, self._cfg, t)
        spot = float(state.get("spot") or 0.0)
        chain = list(state.get("candidate_options") or [])
        return MarketSnapshot(time=t, underlying=self.underlying, spot=spot, option_chain=chain)

    def fill_model(self) -> FillModel:
        return self._fill

    def meta(self) -> Dict[str, Any]:
        return {
            "type": "synthetic_replay",
            "underlying": self.underlying,
            "cfg_class": self._cfg.__class__.__name__,
        }
