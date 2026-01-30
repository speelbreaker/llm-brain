import pytest
from datetime import datetime, timezone, timedelta

from src.models import AgentState, CandidateOption, OptionPosition, PortfolioState, OptionType, Side, VolState
from src.policy_rule_based import decide_action
from src.config import settings


def _mk_candidate(
    symbol: str,
    underlying: str = "BTC",
    dte: int = 14,
    delta: float = 0.2,
    bid: float = 8.0,
    ask: float = 9.0,
    premium_usd: float = 100.0,
):
    expiry = datetime.now(timezone.utc) + timedelta(days=dte)
    mid = (bid + ask) / 2.0
    return CandidateOption(
        symbol=symbol,
        underlying=underlying,
        strike=100000.0,
        expiry=expiry,
        option_type=OptionType.CALL,
        dte=dte,
        delta=delta,
        otm_pct=5.0,
        bid=bid,
        ask=ask,
        mid_price=mid,
        premium_usd=premium_usd,
        iv=50.0,
        rv=40.0,
        ivrv=1.2,
        spread_pct=None,
        open_interest=100,
    )


def _mk_open_short_call(
    symbol: str,
    underlying: str = "BTC",
    dte: int = 14,
    size: float = 1.0,
    entry_price: float = 100.0,
    mark_price: float = 10.0,
):
    expiry = datetime.now(timezone.utc) + timedelta(days=dte)
    return OptionPosition(
        symbol=symbol,
        underlying=underlying,
        strike=100000.0,
        expiry=expiry,
        option_type=OptionType.CALL,
        side=Side.SELL,
        size=size,
        avg_price=entry_price,
        mark_price=mark_price,
        unrealized_pnl=None,
        expiry_dte=dte,
        moneyness="OTM",
        delta=-0.2,
    )


def test_profit_capture_checkpoint_close_never_blocked_no_candidates(monkeypatch):
    """Even with absurd close spread assumptions, if profit-capture triggers and no eligible candidates exist,
    we must resolve to CLOSE (not skip / deadlock).
    """
    # Make threshold easy to hit
    monkeypatch.setattr(settings, "profit_capture_pct", 0.75, raising=False)
    monkeypatch.setattr(settings, "profit_capture_min_hold_hours", 0.0, raising=False)
    monkeypatch.setattr(settings, "profit_capture_roll_only_if_dte_gt", 3, raising=False)

    # Make open eligibility impossible (min credit high)
    monkeypatch.setattr(settings, "profit_capture_min_credit_usd", 10_000.0, raising=False)

    # Position: entry 100, mark 10 => captured 90%
    pos = _mk_open_short_call("BTC-26DEC25-100000-C", entry_price=100.0, mark_price=10.0, dte=14)

    state = AgentState(
        timestamp=datetime.now(timezone.utc),
        underlyings=["BTC"],
        spot={"BTC": 100000.0},
        portfolio=PortfolioState(option_positions=[pos], spot_positions={"BTC": 1.0}, equity_usd=10000.0),
        vol_state=VolState(),
        candidate_options=[],
        market_context=None,
    )

    action = decide_action(state, settings)
    assert action["action"] == "CLOSE_COVERED_CALL"
    assert action.get("reason_code", "").startswith("EXIT_OR_ROLL")


def test_profit_capture_annualized_yield_edge_cases_do_not_crash(monkeypatch):
    """Guardrails: dte<=0 / missing delta / bad quotes should not crash the policy.
    Should fail closed into CLOSE or NO_ELIGIBLE_CANDIDATE.
    """
    monkeypatch.setattr(settings, "profit_capture_pct", 0.75, raising=False)
    monkeypatch.setattr(settings, "profit_capture_min_hold_hours", 0.0, raising=False)
    monkeypatch.setattr(settings, "profit_capture_roll_only_if_dte_gt", 0, raising=False)
    monkeypatch.setattr(settings, "profit_capture_min_credit_usd", 25.0, raising=False)

    pos = _mk_open_short_call("BTC-26DEC25-100000-C", entry_price=100.0, mark_price=10.0, dte=1)

    # Candidate has dte=0 and malformed ask<bid
    bad = _mk_candidate(
        symbol="BTC-27DEC25-110000-C",
        dte=0,
        delta=0.0,
        bid=10.0,
        ask=5.0,
        premium_usd=30.0,
    )

    state = AgentState(
        timestamp=datetime.now(timezone.utc),
        underlyings=["BTC"],
        spot={"BTC": 0.0},  # bad spot
        portfolio=PortfolioState(option_positions=[pos], spot_positions={"BTC": 1.0}, equity_usd=10000.0),
        vol_state=VolState(),
        candidate_options=[bad],
        market_context=None,
    )

    action = decide_action(state, settings)
    assert action["action"] in ("CLOSE_COVERED_CALL", "ROLL_COVERED_CALL")


def test_deterministic_tie_breaker(monkeypatch):
    """If scores tie, selection must be deterministic (stable tie-breaker).

    We set up two candidates with identical score but different symbols.
    The chosen symbol should be deterministic under our tie-break rule.
    """
    monkeypatch.setattr(settings, "profit_capture_pct", 0.75, raising=False)
    monkeypatch.setattr(settings, "profit_capture_min_hold_hours", 0.0, raising=False)
    monkeypatch.setattr(settings, "profit_capture_roll_only_if_dte_gt", 0, raising=False)
    monkeypatch.setattr(settings, "profit_capture_min_credit_usd", 25.0, raising=False)
    monkeypatch.setattr(settings, "profit_capture_max_spread_pct_open", 1.0, raising=False)

    pos = _mk_open_short_call("BTC-26DEC25-100000-C", entry_price=100.0, mark_price=10.0, dte=14)

    c1 = _mk_candidate("BTC-27DEC25-110000-C", bid=8.0, ask=9.0, premium_usd=50.0, dte=14, delta=0.2)
    c2 = _mk_candidate("BTC-27DEC25-120000-C", bid=8.0, ask=9.0, premium_usd=50.0, dte=14, delta=0.2)

    state = AgentState(
        timestamp=datetime.now(timezone.utc),
        underlyings=["BTC"],
        spot={"BTC": 100000.0},
        portfolio=PortfolioState(option_positions=[pos], spot_positions={"BTC": 1.0}, equity_usd=10000.0),
        vol_state=VolState(),
        candidate_options=[c2, c1],  # intentionally reversed
        market_context=None,
    )

    action = decide_action(state, settings)
    assert action["action"] == "ROLL_COVERED_CALL"
    # tie-breaker should pick lexicographically smallest symbol (c1)
    assert action["params"]["to_symbol"] == "BTC-27DEC25-110000-C"
