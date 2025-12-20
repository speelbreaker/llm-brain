from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
import json
import os
from pathlib import Path
import random
from typing import Iterator, List, Optional, Protocol, Literal, Dict, Any, Union

from src.backtest.types import OptionSnapshot

from .types import MarketSnapshot as FidelityMarketSnapshot, OptionQuote


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


def detect_live_dataset(*, underlying: str = "BTC") -> Dict[str, Any]:
    """Best-effort search for harvested live snapshots on disk.

    This intentionally avoids raising: deployed environments vary.
    """
    u = (underlying or "BTC").upper().strip()
    candidates = [
        Path(os.getenv("HARVESTER_DATA_ROOT", "data/live_deribit")),
        Path("data/live_deribit"),
        Path("datasets/live_deribit"),
        Path("storage/live_deribit"),
    ]

    for base in candidates:
        if not base.exists():
            continue
        # Prefer linear USDC folder if present.
        for sub in [f"{u}_USDC", u]:
            p = base / sub
            if p.exists() and p.is_dir():
                return {"found": True, "base_dir": str(base), "underlying_dir": sub}
    return {"found": False}


def load_fixture_snapshots_jsonl(
    path: str,
    *,
    underlying: str,
    start_ts: int,
    end_ts: int,
) -> List[FidelityMarketSnapshot]:
    snaps: List[FidelityMarketSnapshot] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if (obj.get("underlying") or "").upper() != underlying.upper():
                continue
            ts = int(obj.get("ts") or 0)
            if ts < int(start_ts) or ts > int(end_ts):
                continue
            opts = []
            for q in obj.get("options") or []:
                opts.append(
                    OptionQuote(
                        instrument_name=str(q.get("instrument_name")),
                        option_type=str(q.get("option_type")),
                        strike=float(q.get("strike") or 0.0),
                        expiry_ts=int(q.get("expiry_ts") or 0),
                        mark_price=float(q.get("mark_price") or 0.0),
                        mark_iv=(float(q.get("mark_iv")) if q.get("mark_iv") is not None else None),
                        delta=(float(q.get("delta")) if q.get("delta") is not None else None),
                        bid=(float(q.get("bid")) if q.get("bid") is not None else None),
                        ask=(float(q.get("ask")) if q.get("ask") is not None else None),
                    )
                )
            snaps.append(
                FidelityMarketSnapshot(
                    ts=ts,
                    underlying=underlying.upper(),
                    spot=(float(obj.get("spot")) if obj.get("spot") is not None else None),
                    options=opts,
                )
            )

    snaps.sort(key=lambda s: s.ts)
    return snaps


class MarketReplayV2(Protocol):
    underlying: str
    def iter_snapshots(self, start_ts: int, end_ts: int) -> Iterator[FidelityMarketSnapshot]: ...
    def meta(self) -> Dict[str, Any]: ...


def _as_fidelity_snapshot(ms: MarketSnapshot) -> FidelityMarketSnapshot:
    opts: List[OptionQuote] = []
    for o in ms.option_chain:
        opts.append(
            OptionQuote(
                instrument_name=str(getattr(o, "instrument_name", "")),
                option_type=str(getattr(o, "option_type", "")),
                strike=float(getattr(o, "strike", 0.0) or 0.0),
                expiry_ts=int(getattr(o, "expiry_ts", 0) or 0),
                mark_price=float(getattr(o, "mark_price", 0.0) or 0.0),
                mark_iv=(float(getattr(o, "mark_iv", 0.0)) if getattr(o, "mark_iv", None) is not None else None),
                delta=(float(getattr(o, "delta", 0.0)) if getattr(o, "delta", None) is not None else None),
                bid=(float(getattr(o, "best_bid_price", 0.0)) if getattr(o, "best_bid_price", None) is not None else None),
                ask=(float(getattr(o, "best_ask_price", 0.0)) if getattr(o, "best_ask_price", None) is not None else None),
            )
        )
    return FidelityMarketSnapshot(
        ts=int(ms.time.replace(tzinfo=timezone.utc).timestamp()),
        underlying=str(ms.underlying),
        spot=float(ms.spot or 0.0),
        options=opts,
    )


class _LiveReplayAdapter(MarketReplayV2):
    def __init__(self, inner: LiveReplayMarket):
        self.underlying = inner.underlying
        self._inner = inner

    def iter_snapshots(self, start_ts: int, end_ts: int) -> Iterator[FidelityMarketSnapshot]:
        # MVP schedule: one snapshot per UTC day at 00:00.
        start = datetime.fromtimestamp(int(start_ts), tz=timezone.utc)
        end = datetime.fromtimestamp(int(end_ts), tz=timezone.utc)
        t = datetime(start.year, start.month, start.day, tzinfo=timezone.utc)
        while t <= end:
            if t.timestamp() >= start_ts:
                yield _as_fidelity_snapshot(self._inner.snapshot(t))
            t = t + timedelta(days=1)

    def meta(self) -> Dict[str, Any]:
        return self._inner.meta()


class _SyntheticReplayAdapter(MarketReplayV2):
    def __init__(self, inner: SyntheticReplayMarket):
        self.underlying = inner.underlying
        self._inner = inner

    def iter_snapshots(self, start_ts: int, end_ts: int) -> Iterator[FidelityMarketSnapshot]:
        start = datetime.fromtimestamp(int(start_ts), tz=timezone.utc)
        end = datetime.fromtimestamp(int(end_ts), tz=timezone.utc)
        t = datetime(start.year, start.month, start.day, tzinfo=timezone.utc)
        while t <= end:
            if t.timestamp() >= start_ts:
                yield _as_fidelity_snapshot(self._inner.snapshot(t))
            t = t + timedelta(days=1)

    def meta(self) -> Dict[str, Any]:
        return self._inner.meta()


class _FixtureReplay(MarketReplayV2):
    def __init__(self, *, snaps: List[FidelityMarketSnapshot], meta: Dict[str, Any]):
        self._snaps = snaps
        self.underlying = snaps[0].underlying if snaps else "BTC"
        self._meta = meta

    def iter_snapshots(self, start_ts: int, end_ts: int) -> Iterator[FidelityMarketSnapshot]:
        for s in self._snaps:
            if int(start_ts) <= s.ts <= int(end_ts):
                yield s

    def meta(self) -> Dict[str, Any]:
        return self._meta


class _FixtureSynthReplay(MarketReplayV2):
    def __init__(self, *, live_snaps: List[FidelityMarketSnapshot], seed: int, meta: Dict[str, Any]):
        self._live = live_snaps
        self.underlying = live_snaps[0].underlying if live_snaps else "BTC"
        self._seed = int(seed)
        self._meta = meta

    def iter_snapshots(self, start_ts: int, end_ts: int) -> Iterator[FidelityMarketSnapshot]:
        rng = random.Random(self._seed)
        for s in self._live:
            if not (int(start_ts) <= s.ts <= int(end_ts)):
                continue
            # Deterministic per-snapshot jitter.
            spot = float(s.spot or 0.0)
            spot = spot * (1.0 + (rng.random() - 0.5) * 0.001)
            opts: List[OptionQuote] = []
            for q in s.options:
                # Deterministic small distortions.
                eps = (rng.random() - 0.5) * 0.01
                opts.append(
                    OptionQuote(
                        instrument_name=q.instrument_name,
                        option_type=q.option_type,
                        strike=q.strike,
                        expiry_ts=q.expiry_ts,
                        mark_price=float(q.mark_price) * (1.0 + eps),
                        mark_iv=(float(q.mark_iv) * (1.0 + eps) if q.mark_iv is not None else None),
                        delta=q.delta,
                        bid=q.bid,
                        ask=q.ask,
                    )
                )
            yield FidelityMarketSnapshot(ts=s.ts, underlying=s.underlying, spot=spot, options=opts)

    def meta(self) -> Dict[str, Any]:
        return {**self._meta, "seed": self._seed}


def make_live_replay(*, underlying: str, detected: Dict[str, Any]) -> MarketReplayV2:
    from src.backtest.live_deribit_data_source import LiveDeribitDataSource

    base_dir = detected.get("base_dir")
    underlying_dir = detected.get("underlying_dir")
    if not base_dir or not underlying_dir:
        raise ValueError("detected live dataset missing base_dir/underlying_dir")

    # We rely on LiveDeribitDataSource which can read the harvested parquets.
    # It needs a date range; for iter_snapshots we call list_option_chain as-of.
    # We use a wide range to avoid tight coupling; ds will filter internally.
    ds = LiveDeribitDataSource(
        underlying=underlying_dir,
        start_date=datetime(1970, 1, 1, tzinfo=timezone.utc).date(),
        end_date=datetime(2100, 1, 1, tzinfo=timezone.utc).date(),
        base_dir=base_dir,
        canonical_underlying=underlying,
    )
    return _LiveReplayAdapter(
        LiveReplayMarket(
            underlying=underlying,
            ds=ds,
        )
    )


def make_synthetic_replay(live: Union[MarketReplayV2, List[FidelityMarketSnapshot]], *, underlying: str, seed: int) -> MarketReplayV2:
    if isinstance(live, list):
        return _FixtureSynthReplay(live_snaps=live, seed=seed, meta={"type": "fixture_synth"})

    # Harvested path: reuse existing SyntheticReplayMarket (build_historical_state).
    # It consumes the same LiveDeribitDataSource behind the live adapter.
    inner_live = getattr(live, "_inner", None)
    ds = getattr(inner_live, "_ds", None) if inner_live else None
    if ds is None:
        raise ValueError("Unable to build synthetic replay: missing live datasource")

    from src.backtest.types import CallSimulationConfig
    from dataclasses import replace

    cfg = CallSimulationConfig(
        underlying=underlying,
        start=datetime.now(timezone.utc),
        end=datetime.now(timezone.utc),
        timeframe="1d",
        decision_interval_bars=1,
        initial_spot_position=1.0,
        contract_size=1.0,
        fee_rate=0.0005,
    )
    cfg = replace(cfg, chain_mode="synthetic_grid", pricing_mode="synthetic_bs")
    return _SyntheticReplayAdapter(
        SyntheticReplayMarket(
            underlying=underlying,
            ds=ds,
            cfg=cfg,
        )
    )



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
