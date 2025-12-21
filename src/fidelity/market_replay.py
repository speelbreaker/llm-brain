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
        expiry_dt = getattr(o, "expiry", None)
        expiry_ts = getattr(o, "expiry_ts", None)
        if expiry_ts is None and expiry_dt is not None:
            try:
                expiry_ts = int((expiry_dt if expiry_dt.tzinfo else expiry_dt.replace(tzinfo=timezone.utc)).timestamp())
            except Exception:
                expiry_ts = 0
        opts.append(
            OptionQuote(
                instrument_name=str(getattr(o, "instrument_name", "")),
                option_type=str(getattr(o, "kind", getattr(o, "option_type", ""))),
                strike=float(getattr(o, "strike", 0.0) or 0.0),
                expiry_ts=int(expiry_ts or 0),
                mark_price=float(getattr(o, "mark_price", 0.0) or 0.0),
                mark_iv=(float(getattr(o, "iv")) if getattr(o, "iv", None) is not None else (float(getattr(o, "mark_iv")) if getattr(o, "mark_iv", None) is not None else None)),
                delta=(float(getattr(o, "delta")) if getattr(o, "delta", None) is not None else None),
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
        # MVP schedule: one snapshot per UTC day.
        # For harvested datasets, 00:00 UTC often has no nearby snapshot; prefer using
        # the dataset's actual snapshot times when available.
        start = datetime.fromtimestamp(int(start_ts), tz=timezone.utc)
        end = datetime.fromtimestamp(int(end_ts), tz=timezone.utc)

        ds = getattr(self._inner, "_ds", None)
        # LiveDeribitDataSource populates _snapshot_times only after loading.
        if ds is not None and getattr(ds, "_snapshot_times", None) is None and hasattr(ds, "_ensure_loaded"):
            try:
                ds._ensure_loaded()  # type: ignore[attr-defined]
            except Exception:
                pass

        snapshot_times = getattr(ds, "_snapshot_times", None) if ds is not None else None
        if snapshot_times:
            # Normalize to UTC datetimes.
            times: List[datetime] = []
            for t in snapshot_times:
                if isinstance(t, datetime):
                    tt = t if t.tzinfo else t.replace(tzinfo=timezone.utc)
                else:
                    continue
                if start <= tt <= end:
                    times.append(tt.astimezone(timezone.utc))

            # One snapshot per UTC day: pick the earliest available snapshot each day.
            by_day: Dict[tuple[int, int, int], datetime] = {}
            for tt in sorted(times):
                key = (tt.year, tt.month, tt.day)
                if key not in by_day:
                    by_day[key] = tt

            for tt in [by_day[k] for k in sorted(by_day.keys())]:
                if int(tt.timestamp()) >= int(start_ts) and int(tt.timestamp()) <= int(end_ts):
                    yield _as_fidelity_snapshot(self._inner.snapshot(tt))
            return

        # Fallback: daily at 00:00 UTC.
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

        ds = getattr(self._inner, "_ds", None)
        if ds is not None and getattr(ds, "_snapshot_times", None) is None and hasattr(ds, "_ensure_loaded"):
            try:
                ds._ensure_loaded()  # type: ignore[attr-defined]
            except Exception:
                pass

        snapshot_times = getattr(ds, "_snapshot_times", None) if ds is not None else None
        if snapshot_times:
            times: List[datetime] = []
            for t0 in snapshot_times:
                if isinstance(t0, datetime):
                    tt = t0 if t0.tzinfo else t0.replace(tzinfo=timezone.utc)
                else:
                    continue
                if start <= tt <= end:
                    times.append(tt.astimezone(timezone.utc))

            by_day: Dict[tuple[int, int, int], datetime] = {}
            for tt in sorted(times):
                key = (tt.year, tt.month, tt.day)
                if key not in by_day:
                    by_day[key] = tt

            for tt in [by_day[k] for k in sorted(by_day.keys())]:
                if int(tt.timestamp()) >= int(start_ts) and int(tt.timestamp()) <= int(end_ts):
                    yield _as_fidelity_snapshot(self._inner.snapshot(tt))
            return

        # Fallback: daily at 00:00 UTC.
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
        """Return a synthetic-priced option chain aligned to the live instrument universe.

        Fidelity parity needs a non-trivial chain (not just candidates). We therefore:
        1) Load the harvested chain from the datasource at time t
        2) Slice it to a broad ATM band (default +/-20% moneyness) and cfg DTE band
        3) Re-price each instrument with a synthetic BS sigma (cfg-driven)
        """
        from datetime import timedelta

        from src.backtest.pricing import (
            bs_call_delta,
            bs_put_price,
            bs_call_price,
            compute_realized_volatility,
            get_sigma_for_option,
        )

        # Spot
        spot_df = self._ds.get_spot_ohlc(
            underlying=self.underlying,
            start=t - timedelta(hours=24),
            end=t,
            timeframe="1h",
        )
        spot = float(spot_df["close"].iloc[-1]) if not spot_df.empty else 0.0

        # Full harvested chain at time t
        all_options: List[OptionSnapshot] = self._ds.list_option_chain(
            underlying=self.underlying,
            as_of=t,
            settlement_ccy=getattr(self._cfg, "option_settlement_ccy", "USDC"),
            margin_type=getattr(self._cfg, "option_margin_type", "linear"),
        )

        # Slice to a broad-but-bounded universe for fidelity.
        min_dte = float(getattr(self._cfg, "min_dte", 1))
        max_dte = float(getattr(self._cfg, "max_dte", 21))
        m_min = float(os.getenv("FIDELITY_CHAIN_MONEYNESS_MIN", "0.80"))
        m_max = float(os.getenv("FIDELITY_CHAIN_MONEYNESS_MAX", "1.20"))

        # RV history for sigma selection.
        rv_lookback = t - timedelta(days=int(getattr(self._cfg, "synthetic_rv_window_days", 30)) + 7)
        rv_df = self._ds.get_spot_ohlc(
            underlying=self.underlying,
            start=rv_lookback,
            end=t,
            timeframe="1d",
        )
        spot_history = []
        if not rv_df.empty:
            for idx, row in rv_df.iterrows():
                spot_history.append((idx if isinstance(idx, datetime) else t, float(row["close"])))

        # Base sigma; skew is applied inside get_sigma_for_option when abs_delta is provided.
        base_rv = compute_realized_volatility(spot_history, t, int(getattr(self._cfg, "synthetic_rv_window_days", 30)))

        priced: List[OptionSnapshot] = []
        for opt in all_options:
            if spot <= 0:
                continue
            strike = float(getattr(opt, "strike", 0.0) or 0.0)
            if strike <= 0:
                continue
            expiry = getattr(opt, "expiry", None)
            if expiry is None:
                continue
            expiry_dt = expiry if getattr(expiry, "tzinfo", None) else expiry.replace(tzinfo=timezone.utc)
            dte_days = (expiry_dt - (t if t.tzinfo else t.replace(tzinfo=timezone.utc))).total_seconds() / 86400.0
            if dte_days <= 0:
                continue
            if dte_days < min_dte or dte_days > max_dte:
                continue

            m = strike / spot
            if m < m_min or m > m_max:
                continue

            kind = str(getattr(opt, "kind", ""))

            # Time fraction in years
            t_years = max((expiry_dt - (t if t.tzinfo else t.replace(tzinfo=timezone.utc))).total_seconds() / (365.0 * 24 * 3600), 1e-6)

            # If the live delta is present, use it for skew; otherwise leave abs_delta None.
            abs_delta = abs(float(opt.delta)) if getattr(opt, "delta", None) is not None else None
            sigma = get_sigma_for_option(
                config=self._cfg,
                spot_history=spot_history,
                as_of=t,
                option_chain=all_options,
                option_mark_iv=(float(getattr(opt, "iv")) if getattr(opt, "iv", None) is not None else None),
                abs_delta=abs_delta,
                skew_source=str(getattr(self._cfg, "skew_source", "none")),
                dte_days=float(dte_days),
            )

            if kind == "call":
                price = bs_call_price(spot, strike, t_years, sigma, float(getattr(self._cfg, "risk_free_rate", 0.0)))
                delta = bs_call_delta(spot, strike, t_years, sigma, float(getattr(self._cfg, "risk_free_rate", 0.0)))
            else:
                price = bs_put_price(spot, strike, t_years, sigma, float(getattr(self._cfg, "risk_free_rate", 0.0)))
                # Put delta approx via call delta - 1 (r=0 is typical in this codebase)
                delta = bs_call_delta(spot, strike, t_years, sigma, float(getattr(self._cfg, "risk_free_rate", 0.0))) - 1.0

            priced.append(
                OptionSnapshot(
                    instrument_name=str(getattr(opt, "instrument_name", "")),
                    underlying=self.underlying,
                    kind="call" if kind == "call" else "put",
                    strike=strike,
                    expiry=expiry_dt,
                    delta=float(delta) if delta is not None else None,
                    iv=float(sigma),
                    mark_price=float(price),
                    settlement_ccy=str(getattr(opt, "settlement_ccy", getattr(self._cfg, "option_settlement_ccy", "USDC"))),
                    margin_type=str(getattr(opt, "margin_type", getattr(self._cfg, "option_margin_type", "linear"))),
                )
            )

        return MarketSnapshot(time=t, underlying=self.underlying, spot=float(spot), option_chain=priced)

    def fill_model(self) -> FillModel:
        return self._fill

    def meta(self) -> Dict[str, Any]:
        return {
            "type": "synthetic_replay",
            "underlying": self.underlying,
            "cfg_class": self._cfg.__class__.__name__,
            "chain_mode": getattr(self._cfg, "chain_mode", None),
            "pricing_mode": getattr(self._cfg, "pricing_mode", None),
        }
