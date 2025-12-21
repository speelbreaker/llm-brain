import os
import logging
from datetime import datetime, timezone
from typing import List

from .models_mvp import DecisionTrace, CandidateSummary, RiskCheckResult
from .trace_store import TraceStore
from src.deribit_client import DeribitClient

logger = logging.getLogger(__name__)

def run_mvp_cycle(run_reason: str = "schedule") -> DecisionTrace:
    store = TraceStore()
    
    # Initialize trace
    trace = DecisionTrace(
        bot_id="mvp-bot-01",
        strategy_id="covered-call-mvp",
        mode="testnet", # Assumption, or config
        run_reason=run_reason,
        underlying="BTC",
        spot=0.0,
        narrative="Starting cycle"
    )
    
    client = None
    
    try:
        # 1. Load State
        try:
            # We assume credentials are in env/settings. If not, this might fail or work for public endpoints.
            client = DeribitClient()
            
            # Get Index Price
            try:
                trace.spot = client.get_index_price("BTC")
            except Exception as e:
                trace.errors.append(f"Index price fetch failed: {str(e)}")
                # If we can't get spot, we can't really trade or filter well.
                # Use a dummy spot for robust fail-through or stop?
                # For MVP, let's stop.
                raise e
            
            # Get Account (if private keys available)
            try:
                account = client.get_account_summary("BTC")
                trace.account_id = str(account.get("id", "unknown"))
            except Exception as e:
                trace.errors.append(f"Account load failed (check creds): {str(e)}")
                
        except Exception as e:
            trace.errors.append(f"Deribit connection/state failed: {str(e)}")
            trace.decision = "BLOCKED"
            trace.narrative = "Could not connect to exchange or fetch state"
            store.append_trace(trace)
            return trace

        # 2. Build Candidates
        try:
            candidates = _fetch_candidates(client, trace.spot)
            trace.candidates = candidates
            
            if candidates:
                trace.chosen = candidates[0]
                trace.narrative = f"Selected {trace.chosen.instrument_name} with score {trace.chosen.score:.2f}"
            else:
                trace.decision = "NO_TRADE"
                trace.narrative = "No candidates found matching criteria"
                store.append_trace(trace)
                return trace
                
        except Exception as e:
            trace.errors.append(f"Candidate generation failed: {str(e)}")
            trace.decision = "BLOCKED"
            store.append_trace(trace)
            return trace

        # 3. Risk Checks
        risk_passed = True
        
        # Check 1: Kill Switch
        kill_switch = os.environ.get("TRADING_KILL_SWITCH", "0") == "1"
        trace.risk_checks.append(RiskCheckResult(
            name="Kill Switch",
            passed=not kill_switch,
            reason="Kill switch active" if kill_switch else "Kill switch inactive",
            metrics={"kill_switch": 1.0 if kill_switch else 0.0}
        ))
        if kill_switch:
            risk_passed = False
            trace.reason_codes.append("KILL_SWITCH")

        # Check 2: Low IV Guard
        iv_threshold = 30.0
        current_iv = trace.chosen.iv if trace.chosen else 0
        iv_check = current_iv > iv_threshold
        trace.risk_checks.append(RiskCheckResult(
            name="Min IV Guard",
            passed=iv_check,
            reason=f"IV {current_iv:.2f} > {iv_threshold}",
            metrics={"iv": current_iv, "threshold": iv_threshold}
        ))
        if not iv_check:
            risk_passed = False
            trace.reason_codes.append("LOW_IV")
            
        if not risk_passed:
            trace.decision = "BLOCKED"
            trace.narrative = "Risk checks failed"
            store.append_trace(trace)
            return trace

        # 4. Order Intent
        trace.decision = "TRADE"
        amount = 0.1 # Fixed size for MVP
        trace.action = {
            "instrument_name": trace.chosen.instrument_name,
            "side": "sell",
            "amount": amount,
            "type": "limit",
            "price": trace.chosen.premium / trace.spot, # Order price in BTC
            "label": "mvp_cc",
            "post_only": True
        }
        
        # 5. Execution
        trading_enabled = os.environ.get("TRADING_ENABLED", "0") == "1"
        
        if trading_enabled:
            try:
                # Real order
                order = client.place_order(
                    instrument_name=trace.action["instrument_name"],
                    side=trace.action["side"],
                    amount=trace.action["amount"],
                    order_type=trace.action["type"],
                    price=trace.action["price"],
                    post_only=trace.action["post_only"],
                    label=trace.action["label"]
                )
                trace.execution = order
                trace.narrative = f"Order placed: {order.get('order', {}).get('order_id')}"
            except Exception as e:
                trace.errors.append(f"Order placement failed: {str(e)}")
                trace.execution = {"error": str(e), "submitted": True}
        else:
            trace.execution = {"submitted": False, "mode": "paper", "reason": "TRADING_ENABLED=0"}
            trace.narrative = "Paper trade (trading disabled)"

    except Exception as e:
        trace.errors.append(f"Unexpected cycle error: {str(e)}")
        trace.narrative = "Crashed during cycle"
        trace.decision = "BLOCKED"
        logger.exception("MVP Cycle Error")

    finally:
        store.append_trace(trace)
        
    return trace

def _fetch_candidates(client: DeribitClient, spot: float) -> List[CandidateSummary]:
    # Fetch option chain for BTC
    try:
        instruments = client.get_instruments("BTC", kind="option", expired=False)
    except Exception as e:
        # Fallback if instrument fetch fails (e.g. connectivity)
        raise e
    
    # Filter for DTE 7-30
    now = datetime.now(timezone.utc)
    candidates = []
    
    # Optimization: Only process a subset to avoid excessive API calls for ticker
    # We first filter by DTE and Strike in memory
    filtered_insts = []
    for inst in instruments:
        try:
            expiry_ts = inst["expiration_timestamp"] / 1000
            expiry_date = datetime.fromtimestamp(expiry_ts, tz=timezone.utc)
            dte = (expiry_date - now).days
            strike = inst["strike"]
            
            if 7 <= dte <= 30 and inst["option_type"] == "call":
                if 1.05 * spot < strike < 1.15 * spot:
                    filtered_insts.append((inst, expiry_date, dte))
        except Exception:
            continue

    # Limit to top 5 candidates to fetch ticker for
    # Just picking the first 5 matching criteria for MVP to save API calls
    for inst, expiry_date, dte in filtered_insts[:5]:
        try:
            ticker = client.get_ticker(inst["instrument_name"])
            greeks = ticker.get("greeks", {}) or {}
            delta = greeks.get("delta", 0)
            iv = ticker.get("mark_iv", 0)
            mark = ticker.get("mark_price", 0) * spot # Deribit options priced in BTC
            
            # Filter Delta 0.15 - 0.35
            if 0.15 <= delta <= 0.35:
                summary = CandidateSummary(
                    instrument_name=inst["instrument_name"],
                    expiry=expiry_date.isoformat(),
                    strike=inst["strike"],
                    delta=delta,
                    dte=dte,
                    mark=mark,
                    premium=ticker.get("bid_price", 0) * spot, # approx USD
                    iv=iv,
                    score=iv * dte, # simplistic score
                    otm_pct=(inst["strike"] / spot) - 1
                )
                candidates.append(summary)
        except Exception:
            continue
            
    # Sort by score desc
    candidates.sort(key=lambda x: x.score, reverse=True)
    return candidates
