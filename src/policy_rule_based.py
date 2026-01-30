"""
Rule-based policy module.
Implements deterministic decision logic for covered call strategy.
Supports research mode with exploration and production mode with strict filtering.
"""
from __future__ import annotations

import math
import random
from datetime import datetime
from typing import Any

from src.config import Settings, settings
from src.models import ActionType, AgentState, CandidateOption, OptionPosition, Side
from src.scoring.candidates import score_option_candidate

def _get_tracker_entry_time_for_symbol(symbol: str) -> datetime | None:
    """Best-effort lookup of entry_time from PositionTracker persisted state."""
    try:
        from src.position_tracker import position_tracker

        payload = position_tracker.get_open_positions_payload(include_sandbox=True) or {}
        for p in payload.get("positions") or []:
            if p.get("symbol") != symbol:
                continue
            ts = p.get("entry_time")
            if not ts:
                return None
            try:
                return datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except Exception:
                return None
        return None
    except Exception:
        return None



def _get_open_covered_calls(
    agent_state: AgentState,
    underlying: str | None = None,
) -> list[OptionPosition]:
    """Get list of open short call positions (covered calls)."""
    covered_calls = []
    
    for pos in agent_state.portfolio.option_positions:
        if pos.side != Side.SELL:
            continue
        
        if pos.option_type.value != "call":
            continue
        
        if underlying and pos.underlying != underlying:
            continue
        
        covered_calls.append(pos)
    
    return covered_calls


def score_candidate(candidate: CandidateOption, cfg: Settings) -> float:
    """
    Score a candidate covered call using the centralized scoring function.
    Higher score = better candidate.
    
    Uses the shared scoring module (src/scoring/candidates.py) to ensure
    consistent scoring across live agent, backtests, and training.
    
    Args:
        candidate: The candidate option to score
        cfg: Settings configuration
    
    Returns:
        Score value (higher is better)
    """
    target_delta = (cfg.effective_delta_min + cfg.effective_delta_max) / 2.0
    target_dte = (cfg.effective_dte_min + cfg.effective_dte_max) / 2.0

    features = {
        "delta": candidate.delta,
        "dte": candidate.dte,
        "premium_usd": candidate.premium_usd,
        "ivrv": candidate.ivrv,
    }
    
    return score_option_candidate(
        features,
        profile="live",
        config_overrides={
            "target_delta": target_delta,
            "target_dte": target_dte,
            "ivrv_min": cfg.effective_ivrv_min,
        },
    )


def choose_candidate_with_exploration(
    candidates: list[CandidateOption],
    cfg: Settings,
) -> tuple[CandidateOption | None, bool]:
    """
    Choose a candidate, with optional exploration in research mode.
    
    Args:
        candidates: List of candidate options
        cfg: Settings configuration
    
    Returns:
        Tuple of (chosen candidate or None, whether this was an exploration choice)
    """
    if not candidates:
        return None, False

    scored = [(score_candidate(c, cfg), c) for c in candidates]
    scored.sort(key=lambda x: x[0], reverse=True)

    best_score, best_candidate = scored[0]

    if not cfg.is_research or cfg.explore_prob <= 0.0:
        return best_candidate, False

    if random.random() < cfg.explore_prob:
        k = max(1, cfg.explore_top_k)
        top_k = scored[:k]
        _, chosen = random.choice(top_k)
        is_exploration = chosen.symbol != best_candidate.symbol
        return chosen, is_exploration

    return best_candidate, False


def _select_best_candidate(
    candidates: list[CandidateOption],
    underlying: str | None = None,
    exclude_symbols: list[str] | None = None,
    config: Settings | None = None,
) -> CandidateOption | None:
    """
    Select the best candidate option for opening a covered call.
    Uses scoring and exploration logic based on mode.
    
    Args:
        candidates: List of candidate options
        underlying: Filter to specific underlying (optional)
        exclude_symbols: Symbols to exclude (optional)
        config: Settings configuration
    
    Returns:
        Best candidate or None if no suitable candidates
    """
    cfg = config or settings
    exclude = set(exclude_symbols or [])
    
    filtered = []
    for c in candidates:
        if underlying and c.underlying != underlying:
            continue
        if c.symbol in exclude:
            continue
        if c.ivrv < cfg.effective_ivrv_min:
            continue
        filtered.append(c)
    
    if not filtered:
        filtered = [
            c for c in candidates
            if (not underlying or c.underlying == underlying)
            and c.symbol not in exclude
        ]
    
    if not filtered:
        return None
    
    chosen, _ = choose_candidate_with_exploration(filtered, cfg)
    return chosen


def _spread_pct(*, bid: float, ask: float, mark: float, floor_usd: float) -> float:
    denom = max(float(mark or 0.0), float(floor_usd or 0.0), 1e-9)
    return max(0.0, float(ask) - float(bid)) / denom


def _annualized_yield_on_notional(*, credit_usd: float, spot_usd: float, size_underlying: float, dte: float) -> float:
    # Dimensionless yield/year on notional (preferred over unitful 'annualized_credit_usd').
    notional = float(spot_usd) * max(float(size_underlying), 1e-9)
    if notional <= 0:
        return 0.0
    if dte <= 0:
        return 0.0
    return (float(credit_usd) / notional) * (365.0 / float(dte))


def _should_roll_position(
    position: OptionPosition,
    agent_state: AgentState,
    config: Settings | None = None,
) -> tuple[bool, str, dict[str, Any]]:
    """
    Determine if a position should be rolled.
    
    Roll conditions:
    1. Profit capture: short option price decayed enough to capture configured fraction of premium
    2. DTE < 1 day (near expiry)
    3. Position is ITM (assignment risk)
    4. Position is ATM with low DTE (assignment/churn risk)
    """
    cfg = config or settings
    
    dte = position.expiry_dte or 0

    # Profit capture checkpoint:
    # Treat as EXIT_OR_ROLL decision checkpoint, not an automatic roll.
    # Use conservative close-cost estimate and deterministic roll-eligibility.
    try:
        if position.side == Side.SELL and position.avg_price and position.avg_price > 0 and position.mark_price is not None:
            credit0 = float(position.avg_price) * abs(float(position.size or 0.0))
            mark = float(position.mark_price)

            # Conservative close cost estimate:
            # Prefer tracker ask if available (ask is conservative for buyback). Otherwise fall back to mark + buffers.
            floor = float(getattr(cfg, "profit_capture_spread_pct_price_floor_usd", 5.0))
            close_spread_cap = float(getattr(cfg, "profit_capture_max_spread_pct_close", 0.25))

            ask_px: float | None = None
            bid_px: float | None = None
            quote_age_ok = False
            try:
                from datetime import timezone
                from src.position_tracker import position_tracker

                payload = position_tracker.get_open_positions_payload(include_sandbox=True) or {}
                max_age = int(getattr(cfg, "profit_capture_quote_max_age_seconds", 180) or 180)
                now = datetime.now(timezone.utc)
                for p in payload.get("positions") or []:
                    if p.get("symbol") != position.symbol:
                        continue
                    ask_px = float(p.get("ask_price") or 0.0) or None
                    bid_px = float(p.get("bid_price") or 0.0) or None
                    qt = p.get("quote_time")
                    if qt:
                        try:
                            qdt = datetime.fromisoformat(str(qt).replace("Z", "+00:00"))
                            age_s = (now - qdt).total_seconds()
                            quote_age_ok = age_s <= max_age
                        except Exception:
                            quote_age_ok = False
                    break
            except Exception:
                ask_px = None
                bid_px = None
                quote_age_ok = False

            if ask_px is not None and ask_px > 0 and quote_age_ok:
                # Apply slippage buffer on top of ask.
                slippage_tax = (float(getattr(cfg, "paper_slippage_bps", 10.0)) / 10_000.0) * max(ask_px, floor)
                close_cost_est = ask_px + slippage_tax
            else:
                # Fallback approximation when no fresh quote available.
                spread_tax = 0.5 * close_spread_cap * max(mark, floor)
                slippage_tax = (float(getattr(cfg, "paper_slippage_bps", 10.0)) / 10_000.0) * max(mark, floor)
                close_cost_est = mark + spread_tax + slippage_tax

            profit_capture_pct = (credit0 - close_cost_est) / max(credit0, 1e-9)

            if profit_capture_pct >= float(getattr(cfg, "profit_capture_pct", 0.75)):
                if dte > int(getattr(cfg, "profit_capture_roll_only_if_dte_gt", 3)):
                    entry_time = _get_tracker_entry_time_for_symbol(position.symbol)
                    if entry_time is None:
                        return False, "Profit capture hit but entry_time unknown (skip)", {"reason_code": "PROFIT_CAPTURE_ENTRY_TIME_UNKNOWN"}
                    from datetime import timezone
                    age_h = (datetime.now(timezone.utc) - entry_time).total_seconds() / 3600.0
                    if age_h >= float(getattr(cfg, "profit_capture_min_hold_hours", 12.0)):
                        meta = {
                            "reason_code": "EXIT_OR_ROLL_PROFIT_CAPTURE",
                            "credit0": credit0,
                            "close_cost_est": close_cost_est,
                            "profit_capture_pct": profit_capture_pct,
                            "dte": dte,
                            "age_h": age_h,
                        }
                        return True, f"Profit capture checkpoint {profit_capture_pct*100:.1f}% (age={age_h:.1f}h, DTE={dte})", meta
    except Exception:
        pass

    if dte < 1:
        return True, f"Near expiry (DTE={dte})", {"reason_code": "NEAR_EXPIRY"}

    if position.moneyness == "ITM":
        if dte <= 2:
            return True, f"ITM with low DTE ({dte} days) - assignment risk", {"reason_code": "ITM_ASSIGNMENT_RISK"}
    
    spot = agent_state.spot.get(position.underlying, 0)
    if spot > 0 and position.strike > 0:
        pct_from_strike = (position.strike - spot) / spot * 100
        if pct_from_strike < 2.0 and dte <= 1:
            return True, f"ATM (only {pct_from_strike:.1f}% OTM) with low DTE", {"reason_code": "ATM_LOW_DTE"}

    return False, "", {}


def _get_open_positions_summary_from_tracker() -> tuple[int, datetime | None]:
    """Best-effort: use local PositionTracker state (incl. healed positions) to:
    - count total open positions
    - find most recent open_time

    This is used for global entry pacing/caps.
    """
    try:
        from datetime import datetime
        from src.position_tracker import position_tracker

        payload = position_tracker.get_open_positions_payload() or {}
        positions = payload.get("positions") or []

        latest: datetime | None = None
        for p in positions:
            ts = p.get("entry_time")
            if not ts:
                continue
            try:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except Exception:
                continue
            if latest is None or dt > latest:
                latest = dt

        return len(positions), latest
    except Exception:
        return 0, None


def decide_action(
    agent_state: AgentState,
    config: Settings | None = None,
) -> dict[str, Any]:
    """
    Decide the next action based on current state using rule-based logic.
    Uses research vs production mode and exploration settings.
    
    Decision flow:
    1. Check for positions that need rolling
    2. If no open covered calls and good candidates exist, open new position
    3. Otherwise, do nothing
    
    Args:
        agent_state: Current agent state
        config: Settings configuration
    
    Returns:
        Dict with keys: action, params, reasoning, mode, policy_version
    """
    cfg = config or settings
    
    for underlying in cfg.underlyings:
        covered_calls = _get_open_covered_calls(agent_state, underlying)
        
        for cc in covered_calls:
            should_roll, roll_reason, roll_meta = _should_roll_position(cc, agent_state, cfg)

            if should_roll:
                reason_code = str(roll_meta.get("reason_code") or "").strip() if isinstance(roll_meta, dict) else ""

                # Profit-capture checkpoint: deterministic eligibility + scoring; roll only if a good candidate exists.
                if reason_code == "EXIT_OR_ROLL_PROFIT_CAPTURE":
                    floor = float(getattr(cfg, "profit_capture_spread_pct_price_floor_usd", 5.0))
                    max_spread_open = float(getattr(cfg, "profit_capture_max_spread_pct_open", 0.10))
                    min_credit_usd = float(getattr(cfg, "profit_capture_min_credit_usd", 25.0))

                    candidates = [
                        c for c in agent_state.candidate_options
                        if c.underlying == underlying and c.symbol != cc.symbol
                    ]

                    spot = float(agent_state.spot.get(underlying) or 0.0)
                    size_u = abs(float(cc.size or 0.0))

                    eligible: list[tuple[float, CandidateOption]] = []
                    reasons_blocked: list[str] = []

                    for c in candidates:
                        mark = float(c.mid_price or 0.0)
                        spr = _spread_pct(bid=c.bid, ask=c.ask, mark=mark, floor_usd=floor)
                        if spr > max_spread_open:
                            reasons_blocked.append("SPREAD_TOO_WIDE")
                            continue
                        if float(c.premium_usd or 0.0) < min_credit_usd:
                            reasons_blocked.append("BELOW_MIN_CREDIT")
                            continue

                        y = _annualized_yield_on_notional(
                            credit_usd=float(c.premium_usd or 0.0),
                            spot_usd=spot,
                            size_underlying=size_u or float(getattr(cfg, "default_order_size", 0.0) or 0.0),
                            dte=float(c.dte or 0.0),
                        )
                        # Simple deterministic score: yield / |delta| (delta small => higher score), clamp delta floor
                        score = y / max(abs(float(c.delta or 0.0)), 0.05)
                        eligible.append((score, c))

                    eligible.sort(key=lambda x: x[0], reverse=True)
                    chosen = eligible[0][1] if eligible else None

                    meta_out = {
                        **(roll_meta if isinstance(roll_meta, dict) else {}),
                        "eligible_candidates_count": len(eligible),
                        "blocked_reasons": sorted(set(reasons_blocked))[:6],
                    }
                    if chosen is not None:
                        score = eligible[0][0]
                        meta_out["chosen"] = {
                            "symbol": chosen.symbol,
                            "strike": chosen.strike,
                            "expiry": chosen.expiry.isoformat() if hasattr(chosen, "expiry") else None,
                            "dte": chosen.dte,
                            "delta": chosen.delta,
                            "bid": chosen.bid,
                            "ask": chosen.ask,
                            "mid": chosen.mid_price,
                            "premium_usd": chosen.premium_usd,
                            "spread_pct": _spread_pct(bid=chosen.bid, ask=chosen.ask, mark=float(chosen.mid_price or 0.0), floor_usd=floor),
                            "annualized_yield": _annualized_yield_on_notional(
                                credit_usd=float(chosen.premium_usd or 0.0),
                                spot_usd=spot,
                                size_underlying=size_u or float(getattr(cfg, "default_order_size", 0.0) or 0.0),
                                dte=float(chosen.dte or 0.0),
                            ),
                            "score": score,
                        }

                        return {
                            "action": ActionType.ROLL_COVERED_CALL.value,
                            "params": {
                                "underlying": underlying,
                                "from_symbol": cc.symbol,
                                "to_symbol": chosen.symbol,
                                "size": cc.size,
                            },
                            "reason_code": "EXIT_OR_ROLL_PROFIT_CAPTURE",
                            "decision_meta": meta_out,
                            "reasoning": (
                                f"EXIT_OR_ROLL checkpoint: {roll_reason}. "
                                f"Eligible={len(eligible)}; chosen={chosen.symbol} score={score:.4f}. "
                                f"min_credit_usd={min_credit_usd}, max_spread_open={max_spread_open:.2f}. "
                                f"Mode={cfg.mode}, policy={cfg.policy_version}."
                            ),
                            "mode": cfg.mode,
                            "policy_version": cfg.policy_version,
                            "decision_source": "rule_based",
                        }

                    # No eligible candidates -> CLOSE
                    return {
                        "action": ActionType.CLOSE_COVERED_CALL.value,
                        "params": {
                            "underlying": underlying,
                            "symbol": cc.symbol,
                            "size": cc.size,
                        },
                        "reason_code": "EXIT_OR_ROLL_NO_CANDIDATE",
                        "decision_meta": meta_out,
                        "reasoning": (
                            f"EXIT_OR_ROLL checkpoint: {roll_reason}. "
                            f"No eligible roll candidates (min_credit_usd={min_credit_usd}, max_spread_open={max_spread_open:.2f}). "
                            f"Mode={cfg.mode}, policy={cfg.policy_version}."
                        ),
                        "mode": cfg.mode,
                        "policy_version": cfg.policy_version,
                        "decision_source": "rule_based",
                    }

                # Non profit-capture roll behavior (existing)
                candidates = [
                    c for c in agent_state.candidate_options
                    if c.underlying == underlying and c.symbol != cc.symbol
                ]

                if candidates:
                    new_candidate, was_exploration = choose_candidate_with_exploration(candidates, cfg)

                    if new_candidate:
                        explore_tag = "Exploratory " if was_exploration else ""
                        return {
                            "action": ActionType.ROLL_COVERED_CALL.value,
                            "params": {
                                "underlying": underlying,
                                "from_symbol": cc.symbol,
                                "to_symbol": new_candidate.symbol,
                                "size": cc.size,
                            },
                            "reasoning": f"{explore_tag}Rolling {cc.symbol}: {roll_reason}. "
                                       f"New position: {new_candidate.symbol} "
                                       f"(DTE={new_candidate.dte}, delta={new_candidate.delta:.2f}, "
                                       f"premium=${new_candidate.premium_usd:.2f}, IVRV={new_candidate.ivrv:.2f}). "
                                       f"Mode={cfg.mode}, policy={cfg.policy_version}.",
                            "mode": cfg.mode,
                            "policy_version": cfg.policy_version,
                            "decision_source": "rule_based",
                        }

                return {
                    "action": ActionType.CLOSE_COVERED_CALL.value,
                    "params": {
                        "underlying": underlying,
                        "symbol": cc.symbol,
                        "size": cc.size,
                    },
                    "reasoning": f"Closing {cc.symbol}: {roll_reason}. "
                               f"No suitable candidates available for rolling. "
                               f"Mode={cfg.mode}, policy={cfg.policy_version}.",
                    "mode": cfg.mode,
                    "policy_version": cfg.policy_version,
                    "decision_source": "rule_based",
                }
    
    # Global non-training gates (applies before considering any new opens)
    # - max open positions total
    # - max new positions per day (24h rolling window)
    open_total, last_open_time = _get_open_positions_summary_from_tracker()
    if not cfg.is_training_on_testnet:
        if open_total >= cfg.max_open_positions_total:
            return {
                "action": ActionType.DO_NOTHING.value,
                "params": {},
                "reasoning": (
                    f"Global cap hit: open_positions_total={open_total} >= {cfg.max_open_positions_total}. "
                    f"No new entries allowed. Mode={cfg.mode}, policy={cfg.policy_version}."
                ),
                "mode": cfg.mode,
                "policy_version": cfg.policy_version,
                "decision_source": "rule_based",
            }

        if cfg.max_new_positions_per_day_total <= 0:
            return {
                "action": ActionType.DO_NOTHING.value,
                "params": {},
                "reasoning": (
                    f"Entry throttle disabled new entries (max_new_positions_per_day_total={cfg.max_new_positions_per_day_total}). "
                    f"Mode={cfg.mode}, policy={cfg.policy_version}."
                ),
                "mode": cfg.mode,
                "policy_version": cfg.policy_version,
                "decision_source": "rule_based",
            }

        if last_open_time is not None:
            from datetime import datetime, timezone, timedelta
            now = datetime.now(timezone.utc)
            # Rolling 24h window; since we only allow 1/day, this is sufficient.
            if now - last_open_time < timedelta(hours=24):
                remaining = timedelta(hours=24) - (now - last_open_time)
                hrs = int(remaining.total_seconds() // 3600)
                mins = int((remaining.total_seconds() % 3600) // 60)
                return {
                    "action": ActionType.DO_NOTHING.value,
                    "params": {},
                    "reasoning": (
                        f"Entry throttle: last open at {last_open_time.isoformat()}. "
                        f"Next allowed in ~{hrs}h {mins}m (max 1 new position per 24h). "
                        f"Mode={cfg.mode}, policy={cfg.policy_version}."
                    ),
                    "mode": cfg.mode,
                    "policy_version": cfg.policy_version,
                    "decision_source": "rule_based",
                }

    for underlying in cfg.underlyings:
        covered_calls = _get_open_covered_calls(agent_state, underlying)
        existing_symbols = {cc.symbol for cc in covered_calls}
        existing_count = len(covered_calls)
        
        # Exclude already-open symbols to avoid duplicates
        candidates = [
            c for c in agent_state.candidate_options
            if c.underlying == underlying and c.symbol not in existing_symbols
        ]
        
        if not candidates:
            continue
        
        # In training mode on testnet: ALWAYS allow opening new positions if candidates exist
        # Only block if we've reached the absolute max training limit
        if cfg.is_training_on_testnet:
            if existing_count >= cfg.max_calls_per_underlying_training:
                continue
            
            remaining_slots = cfg.max_calls_per_underlying_training - existing_count
            
            # In ladder training mode, sort by premium (most aggressive)
            if cfg.training_profile_mode == "ladder":
                candidates.sort(
                    key=lambda c: (c.premium_usd, c.delta),
                    reverse=True,
                )
                chosen = candidates[0]
                was_exploration = False
            else:
                chosen, was_exploration = choose_candidate_with_exploration(candidates, cfg)
            
            if chosen:
                explore_tag = "Exploratory " if was_exploration else ""
                return {
                    "action": ActionType.OPEN_COVERED_CALL.value,
                    "params": {
                        "underlying": underlying,
                        "symbol": chosen.symbol,
                        "size": cfg.default_order_size,
                    },
                    "reasoning": f"[TRAINING] {explore_tag}OPEN_COVERED_CALL on {chosen.symbol}: "
                               f"DTE={chosen.dte}, delta={chosen.delta:.2f}, "
                               f"premium=${chosen.premium_usd:.2f}, IVRV={chosen.ivrv:.2f}. "
                               f"Existing calls for {underlying}: {existing_count}, "
                               f"remaining training slots: {remaining_slots}. "
                               f"Mode={cfg.mode}, policy={cfg.policy_version}, "
                               f"profile_mode={cfg.training_profile_mode}.",
                    "mode": cfg.mode,
                    "policy_version": cfg.policy_version,
                    "decision_source": "rule_based",
                }
        else:
            # Non-training mode: only open if no existing calls for this underlying
            if covered_calls:
                continue
            
            chosen, was_exploration = choose_candidate_with_exploration(candidates, cfg)
            
            if chosen:
                explore_tag = "Exploratory " if was_exploration else ""
                return {
                    "action": ActionType.OPEN_COVERED_CALL.value,
                    "params": {
                        "underlying": underlying,
                        "symbol": chosen.symbol,
                        "size": cfg.default_order_size,
                    },
                    "reasoning": f"{explore_tag}OPEN_COVERED_CALL on {chosen.symbol}: "
                               f"DTE={chosen.dte}, delta={chosen.delta:.2f}, "
                               f"premium=${chosen.premium_usd:.2f}, IVRV={chosen.ivrv:.2f}. "
                               f"Mode={cfg.mode}, policy={cfg.policy_version}.",
                    "mode": cfg.mode,
                    "policy_version": cfg.policy_version,
                    "decision_source": "rule_based",
                }
    
    existing_positions = []
    for underlying in cfg.underlyings:
        ccs = _get_open_covered_calls(agent_state, underlying)
        existing_positions.extend([cc.symbol for cc in ccs])
    
    if cfg.is_training_on_testnet:
        # In training mode, we only reach here if ALL underlyings are at max positions
        # or there are no valid candidates left
        all_at_max = True
        for underlying in cfg.underlyings:
            ccs = _get_open_covered_calls(agent_state, underlying)
            if len(ccs) < cfg.max_calls_per_underlying_training:
                all_at_max = False
                break
        
        if all_at_max and existing_positions:
            reasoning = (
                f"[TRAINING] All underlyings at max positions ({cfg.max_calls_per_underlying_training}). "
                f"Existing: {', '.join(existing_positions)}. profile_mode={cfg.training_profile_mode}."
            )
        elif not agent_state.candidate_options:
            reasoning = f"[TRAINING] No candidate options available. profile_mode={cfg.training_profile_mode}."
        else:
            reasoning = f"[TRAINING] No new candidates (all symbols open or filtered). profile_mode={cfg.training_profile_mode}."
    elif existing_positions:
        reasoning = f"Existing positions: {', '.join(existing_positions)}. No action needed."
    elif not agent_state.candidate_options:
        reasoning = "No candidate options available that meet criteria."
    else:
        reasoning = "No suitable opportunities identified."
    
    return {
        "action": ActionType.DO_NOTHING.value,
        "params": {},
        "reasoning": f"{reasoning} Mode={cfg.mode}, policy={cfg.policy_version}.",
        "mode": cfg.mode,
        "policy_version": cfg.policy_version,
        "decision_source": "rule_based",
    }
