from __future__ import annotations

from src.fidelity.canonical_strategies import StrategySpec, run_strategy_with_diagnostics
from src.fidelity.types import MarketSnapshot, OptionQuote


def test_strategy_opens_trade_when_delta_missing_via_moneyness_fallback() -> None:
    # Two snapshots: open (with options but delta=None), close (same instrument still present).
    open_ts = 1_700_000_000
    close_ts = open_ts + 3 * 86_400
    expiry_ts = open_ts + 10 * 86_400

    spot = 100.0
    # Create a couple of call strikes; delta missing on purpose.
    opts = [
        OptionQuote(
            instrument_name="BTC-C-110",
            option_type="call",
            strike=110.0,
            expiry_ts=expiry_ts,
            mark_price=5.0,
            bid=4.5,
            ask=5.5,
            mark_iv=None,
            delta=None,
        ),
        OptionQuote(
            instrument_name="BTC-C-112",
            option_type="call",
            strike=112.0,
            expiry_ts=expiry_ts,
            mark_price=4.0,
            bid=3.5,
            ask=4.5,
            mark_iv=None,
            delta=None,
        ),
        OptionQuote(
            instrument_name="BTC-C-120",
            option_type="call",
            strike=120.0,
            expiry_ts=expiry_ts,
            mark_price=2.0,
            bid=1.8,
            ask=2.2,
            mark_iv=None,
            delta=None,
        ),
    ]

    open_snap = MarketSnapshot(ts=open_ts, underlying="BTC", spot=spot, options=list(opts))
    close_snap = MarketSnapshot(ts=close_ts, underlying="BTC", spot=spot, options=list(opts))

    spec = StrategySpec(
        name="CoveredCall",
        entry_target_dte_days=10.0,
        entry_target_abs_delta=0.25,
        hold_days=3,
        kind="short_call",
    )

    trades, diag = run_strategy_with_diagnostics(spec=spec, snapshots=[open_snap, close_snap], slippage_bps=0.0, use_mid=True)
    assert len(trades) >= 1

    # Diagnostics should explain that delta was missing and fallback was used.
    skip = (diag.get("skip_reasons") or {})
    assert skip.get("missing_delta", 0) > 0
    assert skip.get("moneyness_fallback_used", 0) > 0
