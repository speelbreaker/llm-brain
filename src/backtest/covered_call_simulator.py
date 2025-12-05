"""
Core covered call simulation engine.
Answers "what if I sold X call here?" with PnL and drawdown metrics.

Includes:
- Feature extraction for candidate scoring
- Scoring function (0-10) for option attractiveness
- Two exit styles: hold_to_expiry, tp_and_roll
- State-builder integration for historical backtesting
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional, List, Dict, Callable, Any, Literal
from dataclasses import asdict

import numpy as np
import pandas as pd

from .data_source import MarketDataSource
from .types import CallSimulationConfig, SimulatedTrade, SimulationResult, TrainingExample, OptionSnapshot, ExitStyle
from src.models import MarketContext

State = Dict[str, Any]
PolicyFn = Callable[[State], bool]
StateBuilderFn = Callable[[datetime], State]


class CoveredCallSimulator:
    """
    Simple covered-call simulator.

    - simulate_single_call: 'what if I sold a 7DTE ~0.25delta call at this time?'
    - simulate_policy: loop over many decision times and apply a policy
    - simulate_policy_with_scoring: use scoring-based selection with exit styles
    - generate_training_data: emit (state, action, reward) tuples for ML

    Works with any MarketDataSource implementation (Deribit, CSV, Tardis, etc.)
    """

    def __init__(self, data_source: MarketDataSource, config: CallSimulationConfig):
        self.ds = data_source
        self.cfg = config
    
    def _extract_candidate_features(self, state: State, option: OptionSnapshot) -> Dict[str, Any]:
        """
        Build a numeric feature dict for scoring from:
        - option snapshot (delta, dte, iv, ivrv, strike, mark, etc.)
        - market context (regime, returns, vol, distance from MAs)
        - spot price from state
        """
        cfg = self.cfg
        mc_raw = state.get("market_context") or {}
        if isinstance(mc_raw, MarketContext):
            mc = mc_raw.model_dump()
        elif isinstance(mc_raw, dict):
            mc = mc_raw
        else:
            try:
                mc = asdict(mc_raw)
            except (TypeError, AttributeError):
                mc = {}
        
        spot = state.get("spot")
        as_of = state.get("time", datetime.utcnow())
        
        expiry = option.expiry
        dte = (expiry - as_of).total_seconds() / 86400.0
        
        strike = option.strike
        mark = option.mark_price or 0.0
        delta = abs(float(option.delta)) if option.delta is not None else 0.0
        iv = float(option.iv) if option.iv is not None else 0.0
        
        if spot is None or spot <= 0:
            otm_pct = 0.0
            premium_pct = 0.0
        else:
            otm_pct = (strike / spot - 1.0) * 100.0
            premium_pct = (mark / spot) * 100.0 if mark > 0 else 0.0
        
        rv30 = mc.get("realized_vol_30d") or mc.get("realized_vol_30") or 0.0
        ivrv = (iv / rv30) if (iv > 0 and rv30 > 0) else 1.0
        
        regime_label = mc.get("regime", "sideways")
        if regime_label == "bull":
            regime_num = 1
        elif regime_label == "bear":
            regime_num = -1
        else:
            regime_num = 0
        
        return {
            "delta": delta,
            "dte": dte,
            "iv": iv,
            "ivrv": ivrv,
            "otm_pct": otm_pct,
            "premium_pct": premium_pct,
            "regime": regime_num,
            "return_7d_pct": mc.get("return_7d_pct", 0.0),
            "return_30d_pct": mc.get("return_30d_pct", 0.0),
            "realized_vol_7d": mc.get("realized_vol_7d", 0.0),
            "realized_vol_30d": rv30,
            "pct_from_50d_ma": mc.get("pct_from_50d_ma", 0.0),
            "pct_from_200d_ma": mc.get("pct_from_200d_ma", 0.0),
        }
    
    def _score_candidate(self, features: Dict[str, Any]) -> float:
        """
        Hand-crafted scoring function for option candidates.
        Outputs a score in [0, 10]. Higher = more attractive to short.
        
        Scoring components:
        1. IVRV: reward juicy implied vol vs realized
        2. Delta near target (0.25)
        3. DTE near target (7 days default)
        4. Premium richness
        5. Regime adjustments
        6. Vol regime considerations
        7. Distance from 200D MA
        """
        delta = features["delta"]
        dte = features["dte"]
        ivrv = features["ivrv"]
        otm_pct = features["otm_pct"]
        premium_pct = features["premium_pct"]
        regime = features["regime"]
        ret_7d = features["return_7d_pct"]
        ret_30d = features["return_30d_pct"]
        rv7 = features["realized_vol_7d"]
        pct_from_200 = features["pct_from_200d_ma"]
        
        score = 0.0
        
        ivrv_clamped = max(1.0, min(ivrv, 1.5))
        score += (ivrv_clamped - 1.0) / 0.5 * 3.0
        
        target_delta = self.cfg.target_delta
        delta_diff = abs(delta - target_delta)
        delta_score = max(0.0, 1.0 - delta_diff / 0.10) * 2.0
        score += delta_score
        
        target_dte = self.cfg.target_dte
        dte_diff = abs(dte - target_dte)
        dte_score = max(0.0, 1.0 - dte_diff / 2.0) * 1.5
        score += dte_score
        
        prem_clamped = max(0.0, min(premium_pct, 1.5))
        premium_score = (prem_clamped / 1.5) * 2.0 if premium_pct > 0 else 0.0
        score += premium_score
        
        if regime == 1:
            if otm_pct < 5.0:
                score -= 1.0
            if ret_30d > 25.0:
                score -= 0.5
        elif regime == -1:
            if ret_7d < -10.0:
                score -= 0.5
        
        if rv7 > 0 and rv7 < 0.3:
            score -= 0.5
        
        if pct_from_200 > 20.0:
            score -= 0.5
        elif pct_from_200 < -20.0:
            score -= 0.5
        
        return max(0.0, min(score, 10.0))

    @staticmethod
    def _timeframe_to_timedelta(tf: str) -> timedelta:
        """Convert timeframe string to timedelta."""
        if tf == "1m":
            return timedelta(minutes=1)
        if tf == "5m":
            return timedelta(minutes=5)
        if tf == "15m":
            return timedelta(minutes=15)
        if tf == "1h":
            return timedelta(hours=1)
        if tf == "4h":
            return timedelta(hours=4)
        if tf == "1d":
            return timedelta(days=1)
        raise ValueError(f"Unsupported timeframe: {tf}")

    def _generate_decision_times(self) -> List[datetime]:
        """
        Generate decision timestamps between cfg.start and cfg.end
        using timeframe * decision_interval_bars.
        Stop early if target_dte would push expiry beyond cfg.end.
        """
        tf_step = self._timeframe_to_timedelta(self.cfg.timeframe) * self.cfg.decision_interval_bars
        t = self.cfg.start
        decision_times: List[datetime] = []

        max_t = self.cfg.end - timedelta(days=self.cfg.target_dte)

        while t <= max_t:
            decision_times.append(t)
            t += tf_step

        return decision_times

    def _build_state(self, as_of: datetime) -> State:
        """
        Build a simple state representation for a decision time.
        Includes: time, underlying, spot price.
        Can be enriched with IV, realized vol, etc.
        """
        spot_df = self.ds.get_spot_ohlc(
            underlying=self.cfg.underlying,
            start=as_of - self._timeframe_to_timedelta(self.cfg.timeframe),
            end=as_of,
            timeframe=self.cfg.timeframe,
        )
        if spot_df.empty:
            spot_price = None
        else:
            spot_price = float(spot_df["close"].iloc[-1])

        return {
            "time": as_of,
            "underlying": self.cfg.underlying,
            "spot": spot_price,
        }

    def _find_target_call(self, as_of: datetime) -> Optional[OptionSnapshot]:
        """
        Find a call option for cfg.underlying with:
        - expiry ~ target_dte days from as_of (within tolerance),
        - delta ~ target_delta (within tolerance),
        and return the closest delta.
        """
        chain = self.ds.list_option_chain(self.cfg.underlying, as_of=as_of)
        if not chain:
            return None

        target_dte = self.cfg.target_dte
        dte_tol = self.cfg.dte_tolerance
        target_delta = self.cfg.target_delta
        delta_tol = self.cfg.delta_tolerance

        candidates: List[OptionSnapshot] = []
        for opt in chain:
            if opt.kind != "call":
                continue
            if opt.expiry <= as_of:
                continue

            dte_days = (opt.expiry - as_of).total_seconds() / 86400.0
            if abs(dte_days - target_dte) > dte_tol:
                continue

            if opt.delta is None:
                continue

            delta_val = float(opt.delta)
            if abs(delta_val - target_delta) > delta_tol:
                continue

            candidates.append(opt)

        if not candidates:
            return None

        best = min(candidates, key=lambda o: abs(float(o.delta or 0) - target_delta))
        return best

    def simulate_single_call(
        self,
        decision_time: datetime,
        size: float,
    ) -> Optional[SimulatedTrade]:
        """
        'If I sold size units of a target_dte ~target_delta call at decision_time,
         what would PnL and drawdown have been vs HODL?'

        Returns SimulatedTrade with:
        - pnl: covered call portfolio PnL
        - pnl_vs_hodl: incremental PnL vs pure spot holding
        - max_drawdown_pct: maximum drawdown during trade life
        """
        target = self._find_target_call(as_of=decision_time)
        if target is None or target.mark_price is None:
            return None

        cfg = self.cfg
        ds = self.ds

        open_time = decision_time
        open_price = float(target.mark_price)

        spot_df = ds.get_spot_ohlc(
            underlying=cfg.underlying,
            start=decision_time,
            end=target.expiry,
            timeframe=cfg.timeframe,
        )
        if spot_df.empty:
            return None

        opt_df = ds.get_option_ohlc(
            instrument_name=target.instrument_name,
            start=decision_time,
            end=target.expiry,
            timeframe=cfg.timeframe,
        )
        if opt_df.empty:
            return None

        idx = spot_df.index.union(opt_df.index).sort_values()
        spot = spot_df.reindex(idx).ffill()["close"]
        opt_price = opt_df.reindex(idx).ffill()["close"]

        portfolio_values: List[float] = []
        hodl_values: List[float] = []

        for ts in idx:
            s = float(spot.loc[ts])
            c = float(opt_price.loc[ts])

            hodl_val = size * s
            cc_val = size * s + size * (open_price - c)
            portfolio_values.append(cc_val)
            hodl_values.append(hodl_val)

        if not portfolio_values:
            return None

        start_portfolio = portfolio_values[0]
        final_cc = portfolio_values[-1]
        final_hodl = hodl_values[-1]
        final_opt_price = float(opt_price.iloc[-1])

        pnl = final_cc - start_portfolio
        pnl_vs_hodl = final_cc - final_hodl

        peak = portfolio_values[0]
        max_dd_pct = 0.0
        for v in portfolio_values:
            if v > peak:
                peak = v
            dd = (peak - v) / peak if peak > 0 else 0.0
            if dd > max_dd_pct:
                max_dd_pct = dd

        return SimulatedTrade(
            instrument_name=target.instrument_name,
            underlying=cfg.underlying,
            side="SHORT_CALL",
            size=size,
            open_time=open_time,
            close_time=target.expiry,
            open_price=open_price,
            close_price=final_opt_price,
            pnl=pnl,
            pnl_vs_hodl=pnl_vs_hodl,
            max_drawdown_pct=max_dd_pct * 100.0,
            notes=(
                f"target_dte={cfg.target_dte}, target_delta={cfg.target_delta}, "
                f"actual_delta={target.delta}, expiry={target.expiry.isoformat()}"
            ),
        )

    def simulate_policy(
        self,
        policy: PolicyFn,
        size: Optional[float] = None,
    ) -> SimulationResult:
        """
        Loop over many decision times between cfg.start and cfg.end.
        For each decision time:
          - Build a simple state snapshot.
          - Ask policy(state) -> True/False (SELL_CALL vs DO_NOTHING).
          - If True, simulate a single covered call trade from that time.
        Aggregate PnL at trade close to build equity curve.
        """
        trade_size = size if size is not None else self.cfg.initial_spot_position

        decision_times = self._generate_decision_times()
        trades: List[SimulatedTrade] = []
        equity_curve: Dict[datetime, float] = {}
        equity_vs_hodl: Dict[datetime, float] = {}

        cumulative_pnl = 0.0
        cumulative_pnl_vs_hodl = 0.0

        for t in decision_times:
            state = self._build_state(as_of=t)

            if state["spot"] is None:
                continue

            take_trade = policy(state)
            if not take_trade:
                continue

            trade = self.simulate_single_call(decision_time=t, size=trade_size)
            if trade is None:
                continue

            trades.append(trade)

            cumulative_pnl += trade.pnl
            cumulative_pnl_vs_hodl += trade.pnl_vs_hodl

            close_ts = trade.close_time
            equity_curve[close_ts] = cumulative_pnl
            equity_vs_hodl[close_ts] = cumulative_pnl_vs_hodl

        if equity_curve:
            equity_series = pd.Series(equity_curve).sort_index()
            peak = equity_series.cummax()
            dd = (peak - equity_series) / peak.replace(0, float("nan"))
            max_dd = float(dd.max() * 100.0) if not dd.isna().all() else 0.0
            final_equity = float(equity_series.iloc[-1])
        else:
            max_dd = 0.0
            final_equity = 0.0

        avg_pnl = sum(t.pnl for t in trades) / len(trades) if trades else 0.0
        avg_pnl_vs_hodl = sum(t.pnl_vs_hodl for t in trades) / len(trades) if trades else 0.0

        metrics = {
            "num_trades": len(trades),
            "final_pnl": final_equity,
            "avg_pnl": avg_pnl,
            "avg_pnl_vs_hodl": avg_pnl_vs_hodl,
            "max_drawdown_pct": max_dd,
            "win_rate": sum(1 for t in trades if t.pnl > 0) / len(trades) if trades else 0.0,
        }

        return SimulationResult(
            trades=trades,
            equity_curve=equity_curve,
            equity_vs_hodl=equity_vs_hodl,
            metrics=metrics,
        )

    def generate_training_data(
        self,
        policy: Optional[PolicyFn] = None,
    ) -> List[TrainingExample]:
        """
        Generate (state, action, reward) tuples for ML training.

        For each decision time:
        - Build state
        - Determine action (via policy or always trade)
        - Simulate trade to get reward

        Returns list of TrainingExample objects suitable for CSV/JSON export.
        """
        if policy is None:
            policy = lambda state: state.get("spot") is not None

        decision_times = self._generate_decision_times()
        examples: List[TrainingExample] = []

        trade_size = self.cfg.initial_spot_position

        for t in decision_times:
            state = self._build_state(as_of=t)

            if state["spot"] is None:
                continue

            take_trade = policy(state)
            action = "SELL_CALL" if take_trade else "DO_NOTHING"

            if take_trade:
                trade = self.simulate_single_call(decision_time=t, size=trade_size)
                if trade is not None:
                    reward = trade.pnl_vs_hodl
                    extra = {
                        "instrument_name": trade.instrument_name,
                        "pnl": trade.pnl,
                        "pnl_vs_hodl": trade.pnl_vs_hodl,
                        "max_drawdown_pct": trade.max_drawdown_pct,
                    }
                else:
                    reward = 0.0
                    extra = {"error": "no_suitable_option"}
            else:
                reward = 0.0
                extra = {}

            examples.append(
                TrainingExample(
                    decision_time=t,
                    underlying=self.cfg.underlying,
                    spot=state["spot"],
                    action=action,
                    reward=reward,
                    extra=extra,
                )
            )

        return examples
    
    def _simulate_call_hold_to_expiry(
        self,
        decision_time: datetime,
        option_snapshot: OptionSnapshot,
        size: Optional[float] = None,
    ) -> Optional[SimulatedTrade]:
        """
        Simulate a call option held to expiry using a specific option snapshot.
        Used by scoring-based policy for hold_to_expiry exit style.
        """
        cfg = self.cfg
        ds = self.ds
        
        size = size if size is not None else cfg.initial_spot_position
        instrument_name = option_snapshot.instrument_name
        open_price = float(option_snapshot.mark_price or 0.0)
        if open_price <= 0:
            return None
        
        expiry = option_snapshot.expiry
        
        spot_df = ds.get_spot_ohlc(
            underlying=cfg.underlying,
            start=decision_time,
            end=expiry,
            timeframe=cfg.timeframe,
        )
        if spot_df.empty:
            return None
        
        opt_df = ds.get_option_ohlc(
            instrument_name=instrument_name,
            start=decision_time,
            end=expiry,
            timeframe=cfg.timeframe,
        )
        if opt_df.empty:
            return None
        
        idx = spot_df.index.union(opt_df.index).sort_values()
        spot = spot_df.reindex(idx).ffill()["close"]
        opt_price = opt_df.reindex(idx).ffill()["close"]
        
        portfolio_values: List[float] = []
        hodl_values: List[float] = []
        
        for ts in idx:
            s = float(spot.loc[ts])
            c = float(opt_price.loc[ts])
            
            hodl_val = size * s
            cc_val = size * s + size * (open_price - c)
            portfolio_values.append(cc_val)
            hodl_values.append(hodl_val)
        
        if not portfolio_values:
            return None
        
        start_portfolio = portfolio_values[0]
        final_cc = portfolio_values[-1]
        final_hodl = hodl_values[-1]
        final_opt_price = float(opt_price.iloc[-1])
        
        pnl = final_cc - start_portfolio
        pnl_vs_hodl = final_cc - final_hodl
        
        peak = portfolio_values[0]
        max_dd_pct = 0.0
        for v in portfolio_values:
            if v > peak:
                peak = v
            dd = (peak - v) / peak if peak > 0 else 0.0
            if dd > max_dd_pct:
                max_dd_pct = dd
        
        return SimulatedTrade(
            instrument_name=instrument_name,
            underlying=cfg.underlying,
            side="SHORT_CALL",
            size=size,
            open_time=decision_time,
            close_time=expiry,
            open_price=open_price,
            close_price=final_opt_price,
            pnl=pnl,
            pnl_vs_hodl=pnl_vs_hodl,
            max_drawdown_pct=max_dd_pct * 100.0,
            notes=f"exit_style=hold_to_expiry; target_dte={cfg.target_dte}, delta={option_snapshot.delta}",
        )
    
    def _simulate_call_tp_and_roll(
        self,
        decision_time: datetime,
        option_snapshot: OptionSnapshot,
        size: Optional[float] = None,
    ) -> Optional[SimulatedTrade]:
        """
        Simulate a call option with take-profit exit.
        Exit early if option loses TP_THRESHOLD_PCT of its value (e.g., 80%).
        
        This is TP-only for now; roll logic can be added later.
        """
        cfg = self.cfg
        ds = self.ds
        
        size = size if size is not None else cfg.initial_spot_position
        instrument_name = option_snapshot.instrument_name
        open_price = float(option_snapshot.mark_price or 0.0)
        if open_price <= 0:
            return None
        
        expiry = option_snapshot.expiry
        tp_threshold = cfg.tp_threshold_pct / 100.0
        tp_target_price = open_price * (1.0 - tp_threshold)
        
        spot_df = ds.get_spot_ohlc(
            underlying=cfg.underlying,
            start=decision_time,
            end=expiry,
            timeframe=cfg.timeframe,
        )
        if spot_df.empty:
            return None
        
        opt_df = ds.get_option_ohlc(
            instrument_name=instrument_name,
            start=decision_time,
            end=expiry,
            timeframe=cfg.timeframe,
        )
        if opt_df.empty:
            return None
        
        idx = spot_df.index.union(opt_df.index).sort_values()
        spot = spot_df.reindex(idx).ffill()["close"]
        opt_price = opt_df.reindex(idx).ffill()["close"]
        
        portfolio_values: List[float] = []
        hodl_values: List[float] = []
        close_time = expiry
        close_price = open_price
        
        for ts in idx:
            s = float(spot.loc[ts])
            c = float(opt_price.loc[ts])
            
            hodl_val = size * s
            cc_val = size * s + size * (open_price - c)
            portfolio_values.append(cc_val)
            hodl_values.append(hodl_val)
            
            if c <= tp_target_price and ts < expiry:
                close_time = ts
                close_price = c
                break
        
        if not portfolio_values:
            return None
        
        start_portfolio = portfolio_values[0]
        final_cc = portfolio_values[-1]
        final_hodl = hodl_values[-1]
        
        pnl = final_cc - start_portfolio
        pnl_vs_hodl = final_cc - final_hodl
        
        peak = portfolio_values[0]
        max_dd_pct = 0.0
        for v in portfolio_values:
            if v > peak:
                peak = v
            dd = (peak - v) / peak if peak > 0 else 0.0
            if dd > max_dd_pct:
                max_dd_pct = dd
        
        exit_note = "TP_HIT" if close_time < expiry else "EXPIRY"
        
        return SimulatedTrade(
            instrument_name=instrument_name,
            underlying=cfg.underlying,
            side="SHORT_CALL",
            size=size,
            open_time=decision_time,
            close_time=close_time,
            open_price=open_price,
            close_price=close_price,
            pnl=pnl,
            pnl_vs_hodl=pnl_vs_hodl,
            max_drawdown_pct=max_dd_pct * 100.0,
            notes=f"exit_style=tp_and_roll; exit={exit_note}; tp_threshold={cfg.tp_threshold_pct}%",
        )
    
    def simulate_policy_with_scoring(
        self,
        decision_times: List[datetime],
        state_builder: StateBuilderFn,
        exit_style: ExitStyle = "hold_to_expiry",
        min_score_to_trade: Optional[float] = None,
        size: Optional[float] = None,
    ) -> SimulationResult:
        """
        Run a scoring-based policy over many decision times.
        
        Args:
            decision_times: List of datetimes at which to consider trades
            state_builder: Function that builds historical state dict at time t
            exit_style: "hold_to_expiry" or "tp_and_roll"
            min_score_to_trade: Minimum score (0-10) to execute trade
            size: Position size (defaults to cfg.initial_spot_position)
            
        Returns:
            SimulationResult with trades, equity curve, and metrics
        """
        trade_size = size if size is not None else self.cfg.initial_spot_position
        min_score = min_score_to_trade if min_score_to_trade is not None else self.cfg.min_score_to_trade
        
        trades: List[SimulatedTrade] = []
        equity_curve: Dict[datetime, float] = {}
        equity_vs_hodl: Dict[datetime, float] = {}
        
        cumulative_pnl = 0.0
        cumulative_pnl_vs_hodl = 0.0
        
        for t in decision_times:
            state = state_builder(t)
            spot = state.get("spot")
            if spot is None or spot <= 0:
                continue
            
            options = state.get("candidate_options") or []
            if not options:
                continue
            
            scored: List[tuple] = []
            for opt in options:
                feats = self._extract_candidate_features(state, opt)
                s = self._score_candidate(feats)
                scored.append((s, opt, feats))
            
            if not scored:
                continue
            
            scored.sort(key=lambda x: x[0], reverse=True)
            best_score, best_opt, best_feats = scored[0]
            
            if best_score < min_score:
                continue
            
            if exit_style == "hold_to_expiry":
                trade = self._simulate_call_hold_to_expiry(t, best_opt, trade_size)
            else:
                trade = self._simulate_call_tp_and_roll(t, best_opt, trade_size)
            
            if trade is None:
                continue
            
            trades.append(trade)
            cumulative_pnl += trade.pnl
            cumulative_pnl_vs_hodl += trade.pnl_vs_hodl
            
            close_ts = trade.close_time
            equity_curve[close_ts] = cumulative_pnl
            equity_vs_hodl[close_ts] = cumulative_pnl_vs_hodl
        
        if equity_curve:
            idx = pd.DatetimeIndex(sorted(equity_curve.keys()))
            eq = pd.Series([equity_curve[t] for t in idx], index=idx)
            peak = eq.cummax()
            dd = (peak - eq) / peak.replace(0, float("nan"))
            max_dd = float(dd.max() * 100.0) if not dd.isna().all() else 0.0
            final_equity = float(eq.iloc[-1])
        else:
            max_dd = 0.0
            final_equity = 0.0
        
        avg_pnl = sum(t.pnl for t in trades) / len(trades) if trades else 0.0
        avg_pnl_vs_hodl = sum(t.pnl_vs_hodl for t in trades) / len(trades) if trades else 0.0
        
        metrics = {
            "num_trades": len(trades),
            "final_pnl": final_equity,
            "avg_pnl": avg_pnl,
            "avg_pnl_vs_hodl": avg_pnl_vs_hodl,
            "max_drawdown_pct": max_dd,
            "win_rate": sum(1 for t in trades if t.pnl > 0) / len(trades) if trades else 0.0,
            "exit_style": exit_style,
            "min_score_to_trade": min_score,
        }
        
        return SimulationResult(
            trades=trades,
            equity_curve=equity_curve,
            equity_vs_hodl=equity_vs_hodl,
            metrics=metrics,
        )


def always_trade_policy(state: State) -> bool:
    """Simple baseline policy: always sell the call if we have spot data."""
    return state.get("spot") is not None


def never_trade_policy(state: State) -> bool:
    """Policy that never trades - for comparison."""
    return False
