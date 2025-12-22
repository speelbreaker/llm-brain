from __future__ import annotations

from src.fidelity.canonical_strategies import StrategySpec, run_strategy
from src.fidelity.canonical_strategies import run_strategy_with_diagnostics
from src.fidelity.types import MarketSnapshot, OptionQuote


def test_missing_close_quote_invalidates_trade_and_no_fake_profit() -> None:
    # Open snapshot has an option; close snapshot is before expiry but the instrument disappears.
    open_ts = 1_700_000_000
    close_ts = open_ts + 3 * 86_400
    expiry_ts = open_ts + 10 * 86_400

    q = OptionQuote(
        instrument_name="BTC-TEST-1",
        option_type="call",
        strike=100.0,
        expiry_ts=expiry_ts,
        mark_price=10.0,
        bid=9.0,
        ask=11.0,
        delta=0.25,
        mark_iv=0.6,
    )

    open_snap = MarketSnapshot(ts=open_ts, underlying="BTC", spot=100.0, options=[q])
    close_snap = MarketSnapshot(ts=close_ts, underlying="BTC", spot=100.0, options=[])

    spec = StrategySpec(
        name="CoveredCall",
        entry_target_dte_days=10.0,
        entry_target_abs_delta=0.25,
        hold_days=3,
        kind="short_call",
    )

    trades = run_strategy(spec=spec, snapshots=[open_snap, close_snap], slippage_bps=0.0, use_mid=True)
    assert len(trades) == 1

    t = trades[0]
    assert t.is_valid is False
    assert t.data_quality_status == "missing_close_quote"

    # Critical invariant: do NOT treat missing close as price=0, so no fake profit is booked.
    assert t.pnl == 0.0
    assert t.pnl_pct == 0.0


def test_coverage_penalty_forces_untrusted_gate() -> None:
    # This verifies the enforcement logic: if coverage is materially incomplete,
    # the gate must degrade to UNTRUSTED even if other scores look good.
    from src.fidelity.scoring import apply_coverage_penalty, gate_label

    scored = {
        "overall_score": 0.95,
        "component_scores": {"strategy_pnl_parity": 0.95, "underlying_returns": 0.95},
        "redistributed_weights": {"strategy_pnl_parity": 1.0, "underlying_returns": 0.0},
    }

    out = apply_coverage_penalty(scored, coverage_ratio=0.5, invalid_trades_missing_quote=1)
    assert out["component_scores"]["strategy_pnl_parity"] < scored["component_scores"]["strategy_pnl_parity"]

    gate = gate_label(
        overall_score=float(out["overall_score"]),
        coverage_ratio=0.5,
        invalid_trades_missing_quote=1,
    )
    assert gate == "UNTRUSTED"


def test_missing_close_forces_untrusted_gate_even_if_score_high() -> None:
    from src.fidelity.scoring import gate_label

    gate = gate_label(
        overall_score=99.0,
        coverage_ratio=1.0,
        invalid_trades_missing_quote=0,
        invalid_trades_missing_close=1,
    )
    assert gate == "UNTRUSTED"


def test_moneyness_fallback_opens_trade_when_delta_missing() -> None:
    open_ts = 1_700_000_000
    close_ts = open_ts + 3 * 86_400
    expiry_ts = open_ts + 10 * 86_400

    # Delta intentionally missing; runner must fall back to moneyness selection.
    q = OptionQuote(
        instrument_name="BTC-TEST-MISSDELTA",
        option_type="call",
        strike=112.0,  # 1.12 moneyness vs spot=100
        expiry_ts=expiry_ts,
        mark_price=10.0,
        bid=9.0,
        ask=11.0,
        delta=None,
        mark_iv=0.6,
    )

    open_snap = MarketSnapshot(ts=open_ts, underlying="BTC", spot=100.0, options=[q])
    # Close snapshot includes the same instrument so trade can close.
    close_q = OptionQuote(
        instrument_name="BTC-TEST-MISSDELTA",
        option_type="call",
        strike=112.0,
        expiry_ts=expiry_ts,
        mark_price=9.0,
        bid=8.5,
        ask=9.5,
        delta=None,
        mark_iv=0.6,
    )
    close_snap = MarketSnapshot(ts=close_ts, underlying="BTC", spot=100.0, options=[close_q])

    spec = StrategySpec(
        name="CoveredCall",
        entry_target_dte_days=10.0,
        entry_target_abs_delta=0.25,
        hold_days=3,
        kind="short_call",
    )

    trades, diag = run_strategy_with_diagnostics(spec=spec, snapshots=[open_snap, close_snap], slippage_bps=0.0, use_mid=True)
    assert len(trades) >= 1
    assert diag["skip_reasons"].get("moneyness_fallback_used", 0) >= 1
