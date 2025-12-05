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
from .types import CallSimulationConfig, SimulatedTrade, SimulationResult, TrainingExample, OptionSnapshot, ExitStyle, ChainData, ChainLeg, RollTrigger
from .pricing import bs_call_price, bs_call_delta, get_synthetic_iv, compute_realized_volatility
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
        self._spot_history_cache: List[tuple] = []
    
    def _get_synthetic_iv(self, as_of: datetime) -> float:
        """Get synthetic implied volatility for pricing."""
        return get_synthetic_iv(self.cfg, self._spot_history_cache, as_of)
    
    def _compute_synthetic_option_price(
        self,
        spot: float,
        strike: float,
        expiry: datetime,
        as_of: datetime,
    ) -> tuple:
        """
        Compute synthetic call option price and delta using Black-Scholes.
        
        Returns:
            Tuple of (price, delta)
        """
        sigma = self._get_synthetic_iv(as_of)
        t_years = max((expiry - as_of).total_seconds() / (365.0 * 24 * 3600), 1e-6)
        r = self.cfg.risk_free_rate
        
        price = bs_call_price(spot, strike, t_years, sigma, r)
        delta = bs_call_delta(spot, strike, t_years, sigma, r)
        
        return price, delta
    
    def _build_spot_history_cache(self, start: datetime, end: datetime) -> None:
        """Pre-fetch spot history for synthetic IV calculation."""
        lookback = timedelta(days=max(60, self.cfg.synthetic_rv_window_days + 10))
        fetch_start = start - lookback
        
        spot_df = self.ds.get_spot_ohlc(
            underlying=self.cfg.underlying,
            start=fetch_start,
            end=end,
            timeframe="1d",
        )
        
        if not spot_df.empty:
            cache_list = []
            for ts, row in spot_df.iterrows():
                ts_dt = ts.to_pydatetime() if hasattr(ts, 'to_pydatetime') else datetime.utcnow()
                cache_list.append((ts_dt, float(row["close"])))
            self._spot_history_cache = cache_list
        else:
            self._spot_history_cache = []
    
    def _generate_synthetic_option_prices(
        self,
        spot_series: pd.Series,
        strike: float,
        expiry: datetime,
    ) -> pd.Series:
        """
        Generate synthetic option prices for an entire spot time series.
        
        Args:
            spot_series: Pandas Series with datetime index and spot prices
            strike: Option strike price
            expiry: Option expiry datetime
        
        Returns:
            Pandas Series with synthetic option prices
        """
        sigma = self._get_synthetic_iv(expiry)
        r = self.cfg.risk_free_rate
        
        prices = []
        for ts, spot_val in spot_series.items():
            ts_dt: datetime
            if hasattr(ts, 'to_pydatetime'):
                ts_dt = ts.to_pydatetime()
            elif isinstance(ts, datetime):
                ts_dt = ts
            else:
                ts_dt = datetime.utcnow()
            
            t_years = max((expiry - ts_dt).total_seconds() / (365.0 * 24 * 3600), 1e-6)
            price = bs_call_price(float(spot_val), strike, t_years, sigma, r)
            prices.append(price)
        
        return pd.Series(prices, index=spot_series.index)
    
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
        
        Supports both pricing modes:
        - deribit_live: Uses actual option OHLC from data source
        - synthetic_bs: Uses Black-Scholes synthetic pricing from spot
        """
        cfg = self.cfg
        ds = self.ds
        use_synthetic = cfg.pricing_mode == "synthetic_bs"
        
        size = size if size is not None else cfg.initial_spot_position
        instrument_name = option_snapshot.instrument_name
        strike = option_snapshot.strike
        expiry = option_snapshot.expiry
        
        spot_df = ds.get_spot_ohlc(
            underlying=cfg.underlying,
            start=decision_time,
            end=expiry,
            timeframe=cfg.timeframe,
        )
        if spot_df.empty:
            return None
        
        if use_synthetic:
            if not self._spot_history_cache:
                self._build_spot_history_cache(cfg.start, cfg.end)
            
            spot_at_open = float(spot_df["close"].iloc[0])
            open_price, _ = self._compute_synthetic_option_price(spot_at_open, strike, expiry, decision_time)
            
            if open_price <= 0:
                return None
            
            idx = spot_df.index.sort_values()
            spot = spot_df.reindex(idx).ffill()["close"]
            opt_price = self._generate_synthetic_option_prices(spot, strike, expiry)
        else:
            open_price = float(option_snapshot.mark_price or 0.0)
            if open_price <= 0:
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
        
        pricing_note = "synthetic_bs" if use_synthetic else "deribit_live"
        
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
            notes=f"exit_style=hold_to_expiry; pricing={pricing_note}; delta={option_snapshot.delta}",
        )
    
    def _simulate_call_tp_and_roll(
        self,
        decision_time: datetime,
        option_snapshot: OptionSnapshot,
        size: Optional[float] = None,
    ) -> Optional[SimulatedTrade]:
        """
        Simulate a multi-roll call chain with TP and defensive roll triggers.
        
        Roll triggers:
        1. Take-profit: When capture_frac >= tp_threshold_pct (e.g., 80%)
        2. Defensive: When spot/strike >= defend_near_strike_pct (e.g., 98%)
        
        Only rolls if DTE > min_dte_to_roll and rolls_used < max_rolls_per_chain.
        
        Supports both pricing modes:
        - deribit_live: Uses actual option OHLC from data source
        - synthetic_bs: Uses Black-Scholes synthetic pricing from spot
        """
        from .state_builder import build_historical_state
        
        cfg = self.cfg
        ds = self.ds
        use_synthetic = cfg.pricing_mode == "synthetic_bs"
        
        if use_synthetic and not self._spot_history_cache:
            self._build_spot_history_cache(cfg.start, cfg.end)
        
        size = size if size is not None else cfg.initial_spot_position
        
        max_rolls = cfg.max_rolls_per_chain
        tp_frac = cfg.tp_threshold_pct / 100.0
        defend_thresh = cfg.defend_near_strike_pct
        min_dte_roll = cfg.min_dte_to_roll
        min_score = cfg.min_score_to_trade
        
        realized_pnl = 0.0
        realized_pnl_vs_hodl = 0.0
        equity_curve: List[float] = []
        hodl_curve: List[float] = []
        legs_count = 0
        rolls_used = 0
        
        chain_legs: List[ChainLeg] = []
        
        first_open_time = decision_time
        last_close_time = decision_time
        first_instrument = option_snapshot.instrument_name
        
        current_opt = option_snapshot
        current_leg_open_time = decision_time
        
        while current_opt is not None:
            instrument_name = current_opt.instrument_name
            strike = current_opt.strike
            expiry = current_opt.expiry
            backtest_end = cfg.end
            sim_end = min(expiry, backtest_end)
            
            spot_df = ds.get_spot_ohlc(
                underlying=cfg.underlying,
                start=current_leg_open_time,
                end=sim_end,
                timeframe=cfg.timeframe,
            )
            if spot_df.empty:
                break
            
            if use_synthetic:
                spot_at_open = float(spot_df["close"].iloc[0])
                open_price, _ = self._compute_synthetic_option_price(spot_at_open, strike, expiry, current_leg_open_time)
                if open_price <= 0:
                    break
                
                idx = spot_df.index.sort_values()
                spot_series = spot_df.reindex(idx).ffill()["close"]
                opt_price_series = self._generate_synthetic_option_prices(spot_series, strike, expiry)
            else:
                open_price = float(current_opt.mark_price or 0.0)
                if open_price <= 0:
                    break
                
                opt_df = ds.get_option_ohlc(
                    instrument_name=instrument_name,
                    start=current_leg_open_time,
                    end=sim_end,
                    timeframe=cfg.timeframe,
                )
                if opt_df.empty:
                    break
                
                idx = spot_df.index.union(opt_df.index).sort_values()
                spot_series = spot_df.reindex(idx).ffill()["close"]
                opt_price_series = opt_df.reindex(idx).ffill()["close"]
            
            leg_close_time = sim_end
            leg_close_price = open_price
            last_observed_time = current_leg_open_time
            last_observed_price = open_price
            roll_triggered = False
            roll_time: Optional[datetime] = None
            
            for ts in idx:
                if ts <= current_leg_open_time:
                    continue
                    
                spot_now = float(spot_series.loc[ts])
                opt_now = float(opt_price_series.loc[ts])
                
                last_observed_time = ts
                last_observed_price = opt_now
                
                dte_now = (expiry - ts).total_seconds() / 86400.0
                
                premium_captured = open_price - opt_now
                capture_frac = premium_captured / open_price if open_price > 0 else 0.0
                
                portfolio_val = (
                    size * spot_now
                    + realized_pnl
                    + size * (open_price - opt_now)
                )
                hodl_val = size * spot_now
                equity_curve.append(portfolio_val)
                hodl_curve.append(hodl_val)
                
                if dte_now <= 0:
                    leg_close_time = ts
                    leg_close_price = opt_now
                    break
                
                if rolls_used < max_rolls and dte_now > min_dte_roll:
                    tp_trigger = capture_frac >= tp_frac
                    defensive_trigger = (spot_now / strike) >= defend_thresh if strike > 0 else False
                    
                    if tp_trigger or defensive_trigger:
                        leg_close_time = ts
                        leg_close_price = opt_now
                        roll_triggered = True
                        roll_time = ts
                        break
            else:
                leg_close_time = last_observed_time
                leg_close_price = last_observed_price
            
            leg_pnl = size * (open_price - leg_close_price)
            leg_hodl_at_close = hodl_curve[-1] if hodl_curve else size * float(spot_series.iloc[-1])
            leg_portfolio_at_close = equity_curve[-1] if equity_curve else leg_pnl
            leg_pnl_vs_hodl = leg_portfolio_at_close - leg_hodl_at_close
            
            dte_at_open = (expiry - current_leg_open_time).total_seconds() / 86400.0
            
            leg_trigger: RollTrigger = "none"
            if roll_triggered:
                premium_captured = open_price - leg_close_price
                capture_frac = premium_captured / open_price if open_price > 0 else 0.0
                if capture_frac >= tp_frac:
                    leg_trigger = "tp_roll"
                else:
                    leg_trigger = "defensive_roll"
            elif leg_close_time >= expiry or (expiry - leg_close_time).total_seconds() / 86400.0 <= 0:
                leg_trigger = "expiry"
            
            chain_legs.append(ChainLeg(
                index=legs_count,
                instrument_name=instrument_name,
                open_time=current_leg_open_time,
                close_time=leg_close_time,
                strike=strike,
                dte_open=dte_at_open,
                open_price=open_price,
                close_price=leg_close_price,
                pnl=leg_pnl,
                trigger=leg_trigger,
            ))
            
            realized_pnl += leg_pnl
            realized_pnl_vs_hodl = leg_portfolio_at_close - leg_hodl_at_close if hodl_curve else 0.0
            legs_count += 1
            last_close_time = leg_close_time
            
            if roll_triggered and roll_time is not None:
                rolls_used += 1
                
                state_roll: Optional[Dict[str, Any]] = None
                candidates: List[OptionSnapshot] = []
                try:
                    from .deribit_data_source import DeribitDataSource as DDS
                    if isinstance(ds, DDS):
                        state_roll = build_historical_state(ds, cfg, roll_time)
                        candidates = state_roll.get("candidate_options") or []
                except Exception:
                    candidates = []
                
                scored = []
                for opt in candidates:
                    if opt.instrument_name == instrument_name:
                        continue
                    try:
                        roll_state = state_roll if state_roll else {"time": roll_time, "spot": None}
                        feats = self._extract_candidate_features(roll_state, opt)
                        s = self._score_candidate(feats)
                        if s >= min_score:
                            scored.append((s, opt))
                    except Exception:
                        continue
                
                if scored:
                    scored.sort(key=lambda x: x[0], reverse=True)
                    _, next_opt = scored[0]
                    current_opt = next_opt
                    current_leg_open_time = roll_time
                else:
                    current_opt = None
            else:
                current_opt = None
        
        if not equity_curve:
            return None
        
        peak = equity_curve[0]
        max_dd_pct = 0.0
        for v in equity_curve:
            if v > peak:
                peak = v
            dd = (peak - v) / peak if peak > 0 else 0.0
            if dd > max_dd_pct:
                max_dd_pct = dd
        
        pricing_note = "synthetic_bs" if use_synthetic else "deribit_live"
        notes = (
            f"multi_roll: {legs_count} legs, rolls_used={rolls_used}, "
            f"tp={tp_frac*100:.0f}%, defend={defend_thresh*100:.0f}%, max_rolls={max_rolls}, "
            f"pricing={pricing_note}"
        )
        
        chain_data = ChainData(
            decision_time=decision_time,
            underlying=cfg.underlying,
            total_pnl=realized_pnl,
            max_drawdown_pct=max_dd_pct * 100.0,
            legs=chain_legs,
        )
        
        return SimulatedTrade(
            instrument_name=first_instrument,
            underlying=cfg.underlying,
            side="SHORT_CALL",
            size=size,
            open_time=first_open_time,
            close_time=last_close_time,
            open_price=float(option_snapshot.mark_price or 0.0),
            close_price=0.0,
            pnl=realized_pnl,
            pnl_vs_hodl=realized_pnl_vs_hodl,
            max_drawdown_pct=max_dd_pct * 100.0,
            notes=notes,
            chain=chain_data,
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
