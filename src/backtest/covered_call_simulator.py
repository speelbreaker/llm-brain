"""
Core covered call simulation engine.
Answers "what if I sold X call here?" with PnL and drawdown metrics.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional, List, Dict, Callable, Any

import pandas as pd

from .deribit_data_source import DeribitDataSource, OptionSnapshot
from .types import CallSimulationConfig, SimulatedTrade, SimulationResult, TrainingExample

State = Dict[str, Any]
PolicyFn = Callable[[State], bool]


class CoveredCallSimulator:
    """
    Simple covered-call simulator.

    - simulate_single_call: 'what if I sold a 7DTE ~0.25delta call at this time?'
    - simulate_policy: loop over many decision times and apply a policy
    - generate_training_data: emit (state, action, reward) tuples for ML
    """

    def __init__(self, data_source: DeribitDataSource, config: CallSimulationConfig):
        self.ds = data_source
        self.cfg = config

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


def always_trade_policy(state: State) -> bool:
    """Simple baseline policy: always sell the call if we have spot data."""
    return state.get("spot") is not None


def never_trade_policy(state: State) -> bool:
    """Policy that never trades - for comparison."""
    return False
