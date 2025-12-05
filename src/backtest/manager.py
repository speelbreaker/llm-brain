"""
Backtest Manager for running backtests with live progress tracking.
Only one active backtest at a time. Supports pause/resume and "both" exit style comparison.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from threading import Thread, Lock, Event
from typing import Any, Dict, List, Literal, Optional


def _sanitize_float(val: float) -> float:
    """Replace inf/nan with JSON-safe values."""
    if math.isnan(val) or math.isinf(val):
        return 0.0
    return val


def _sanitize_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively sanitize dict values for JSON serialization."""
    result = {}
    for k, v in d.items():
        if isinstance(v, float):
            result[k] = _sanitize_float(v)
        elif isinstance(v, dict):
            result[k] = _sanitize_dict(v)
        elif isinstance(v, list):
            result[k] = [
                _sanitize_float(x) if isinstance(x, float) else 
                (_sanitize_dict(x) if isinstance(x, dict) else x)
                for x in v
            ]
        else:
            result[k] = v
    return result


MarginType = Literal["linear", "inverse"]
SettlementCcy = Literal["ANY", "USDC", "BTC", "ETH"]

ExitStyle = Literal["hold_to_expiry", "tp_and_roll", "both"]
RollTrigger = Literal["tp_roll", "defensive_roll", "expiry", "none"]

MAX_RECENT_CHAINS = 50


def _compute_equity_curve(
    trades: List[Any],
    spot_at_times: Dict[datetime, float],
    position_size: float,
    start_time: datetime,
    end_time: datetime,
) -> List["EquityCurvePoint"]:
    """
    Compute equity curve from trades and spot prices.
    
    Args:
        trades: List of SimulatedTrade objects sorted by close_time
        spot_at_times: Dict mapping datetime to spot price
        position_size: Size of underlying position (e.g., 1.0 BTC)
        start_time: Backtest start time
        end_time: Backtest end time
    
    Returns:
        List of EquityCurvePoint objects
    """
    if not spot_at_times:
        return []
    
    sorted_times = sorted(spot_at_times.keys())
    if not sorted_times:
        return []
    
    spot_start = spot_at_times.get(start_time) or spot_at_times.get(sorted_times[0], 0)
    if spot_start <= 0:
        spot_start = spot_at_times.get(sorted_times[0], 10000)
    
    base_equity = spot_start * position_size
    
    points: List[EquityCurvePoint] = []
    
    points.append(EquityCurvePoint(
        time=start_time,
        equity=base_equity,
        hodl_equity=base_equity,
    ))
    
    cumulative_pnl_vs_hodl = 0.0
    
    sorted_trades = sorted(trades, key=lambda t: t.close_time)
    
    for trade in sorted_trades:
        cumulative_pnl_vs_hodl += trade.pnl_vs_hodl
        
        close_time = trade.close_time
        spot_now = spot_at_times.get(close_time)
        if spot_now is None:
            closest_time = min(sorted_times, key=lambda t: abs((t - close_time).total_seconds()))
            spot_now = spot_at_times.get(closest_time, spot_start)
        
        hodl_equity = spot_now * position_size
        equity = hodl_equity + cumulative_pnl_vs_hodl
        
        points.append(EquityCurvePoint(
            time=close_time,
            equity=equity,
            hodl_equity=hodl_equity,
        ))
    
    if sorted_trades:
        last_trade_time = sorted_trades[-1].close_time
        if end_time > last_trade_time:
            spot_end = spot_at_times.get(end_time)
            if spot_end is None:
                closest_time = min(sorted_times, key=lambda t: abs((t - end_time).total_seconds()))
                spot_end = spot_at_times.get(closest_time, spot_start)
            
            hodl_equity = spot_end * position_size
            equity = hodl_equity + cumulative_pnl_vs_hodl
            
            points.append(EquityCurvePoint(
                time=end_time,
                equity=equity,
                hodl_equity=hodl_equity,
            ))
    
    return points


def _compute_enhanced_metrics(
    trades: List[Any],
    equity_curve: List["EquityCurvePoint"],
    position_size: float,
) -> Dict[str, Any]:
    """
    Compute enhanced strategy metrics like TradingView.
    
    Args:
        trades: List of SimulatedTrade objects
        equity_curve: List of EquityCurvePoint objects
        position_size: Size of underlying position
    
    Returns:
        Dict with enhanced metrics
    """
    if not equity_curve:
        return {}
    
    initial_equity = equity_curve[0].equity if equity_curve else 0.0
    final_equity = equity_curve[-1].equity if equity_curve else 0.0
    final_hodl = equity_curve[-1].hodl_equity if equity_curve else 0.0
    
    net_profit_usd = final_equity - initial_equity
    net_profit_pct = (net_profit_usd / initial_equity * 100) if initial_equity > 0 else 0.0
    
    hodl_profit_usd = final_hodl - initial_equity
    hodl_profit_pct = (hodl_profit_usd / initial_equity * 100) if initial_equity > 0 else 0.0
    
    max_drawdown_pct = 0.0
    max_drawdown_usd = 0.0
    peak_equity = equity_curve[0].equity if equity_curve else 0.0
    
    for pt in equity_curve:
        if pt.equity > peak_equity:
            peak_equity = pt.equity
        drawdown_usd = peak_equity - pt.equity
        drawdown_pct = (drawdown_usd / peak_equity * 100) if peak_equity > 0 else 0.0
        if drawdown_pct > max_drawdown_pct:
            max_drawdown_pct = drawdown_pct
            max_drawdown_usd = drawdown_usd
    
    num_trades = len(trades)
    winning_trades = [t for t in trades if t.pnl > 0]
    losing_trades = [t for t in trades if t.pnl <= 0]
    win_count = len(winning_trades)
    win_rate = (win_count / num_trades * 100) if num_trades > 0 else 0.0
    
    total_pnl = sum(t.pnl for t in trades)
    avg_trade_usd = total_pnl / num_trades if num_trades > 0 else 0.0
    
    gross_profit = sum(t.pnl for t in winning_trades)
    gross_loss = abs(sum(t.pnl for t in losing_trades))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (float('inf') if gross_profit > 0 else 0.0)
    
    avg_winner = gross_profit / len(winning_trades) if winning_trades else 0.0
    avg_loser = gross_loss / len(losing_trades) if losing_trades else 0.0
    
    total_pnl_vs_hodl = sum(t.pnl_vs_hodl for t in trades)
    
    pct_returns = []
    if len(equity_curve) > 1:
        for i in range(1, len(equity_curve)):
            prev_equity = equity_curve[i - 1].equity
            curr_equity = equity_curve[i].equity
            if prev_equity > 0:
                ret = (curr_equity - prev_equity) / prev_equity
                pct_returns.append(ret)
    
    mean_return = sum(pct_returns) / len(pct_returns) if pct_returns else 0.0
    
    std_return = 0.0
    if len(pct_returns) > 1:
        variance = sum((r - mean_return) ** 2 for r in pct_returns) / (len(pct_returns) - 1)
        std_return = math.sqrt(variance)
    
    sharpe_ratio = (mean_return / std_return) if std_return > 0 else 0.0
    
    downside_returns = [r for r in pct_returns if r < 0]
    downside_std = 0.0
    if len(downside_returns) > 1:
        downside_variance = sum(r ** 2 for r in downside_returns) / (len(downside_returns) - 1)
        downside_std = math.sqrt(downside_variance)
    elif len(downside_returns) == 1:
        downside_std = abs(downside_returns[0])
    
    sortino_ratio = (mean_return / downside_std) if downside_std > 0 else 0.0
    
    return {
        "initial_equity": round(initial_equity, 2),
        "final_equity": round(final_equity, 2),
        "net_profit_usd": round(net_profit_usd, 2),
        "net_profit_pct": round(net_profit_pct, 2),
        "hodl_profit_usd": round(hodl_profit_usd, 2),
        "hodl_profit_pct": round(hodl_profit_pct, 2),
        "max_drawdown_pct": round(max_drawdown_pct, 2),
        "max_drawdown_usd": round(max_drawdown_usd, 2),
        "num_trades": num_trades,
        "win_rate": round(win_rate, 1),
        "avg_trade_usd": round(avg_trade_usd, 2),
        "profit_factor": round(profit_factor, 2) if profit_factor != float('inf') else 999.99,
        "gross_profit": round(gross_profit, 2),
        "gross_loss": round(gross_loss, 2),
        "avg_winner": round(avg_winner, 2),
        "avg_loser": round(avg_loser, 2),
        "sharpe_ratio": round(sharpe_ratio, 2),
        "sortino_ratio": round(sortino_ratio, 2),
        "final_pnl": round(total_pnl, 4),
        "final_pnl_vs_hodl": round(total_pnl_vs_hodl, 4),
        "avg_pnl": round(avg_trade_usd, 4),
    }


@dataclass
class BacktestProgressStep:
    time: datetime
    candidates: int
    best_score: float
    traded: bool
    exit_style: str


@dataclass
class BacktestLegSummary:
    """Summary of a single leg in a chain."""
    index: int
    open_time: datetime
    close_time: datetime
    strike: float
    dte_open: float
    pnl: float
    trigger: RollTrigger


@dataclass
class BacktestChainSummary:
    """Summary of a multi-leg call chain."""
    decision_time: datetime
    underlying: str
    num_legs: int
    num_rolls: int
    total_pnl: float
    max_drawdown_pct: float
    exit_style: str
    legs: List[BacktestLegSummary] = field(default_factory=list)


@dataclass
class EquityCurvePoint:
    """A single point in the equity curve."""
    time: datetime
    equity: float
    hodl_equity: float


@dataclass
class BacktestStatus:
    running: bool = False
    paused: bool = False
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    progress_pct: float = 0.0
    current_time: Optional[datetime] = None
    decisions_processed: int = 0
    total_decisions: int = 0
    exit_style: Optional[str] = None
    current_phase: Optional[str] = None
    underlying: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    recent_steps: List[BacktestProgressStep] = field(default_factory=list)
    recent_chains: List[BacktestChainSummary] = field(default_factory=list)
    equity_curve: List[EquityCurvePoint] = field(default_factory=list)
    error: Optional[str] = None


class BacktestManager:
    def __init__(self) -> None:
        self._lock = Lock()
        self._thread: Optional[Thread] = None
        self._status = BacktestStatus()
        self._cancel_requested = False
        self._pause_event = Event()
        self._pause_event.set()

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            chains_json = []
            for chain in self._status.recent_chains:
                chains_json.append({
                    "decision_time": chain.decision_time.isoformat(),
                    "underlying": chain.underlying,
                    "num_legs": chain.num_legs,
                    "num_rolls": chain.num_rolls,
                    "total_pnl": _sanitize_float(chain.total_pnl),
                    "max_drawdown_pct": _sanitize_float(chain.max_drawdown_pct),
                    "exit_style": chain.exit_style,
                    "legs": [
                        {
                            "index": leg.index,
                            "open_time": leg.open_time.isoformat(),
                            "close_time": leg.close_time.isoformat(),
                            "strike": _sanitize_float(leg.strike),
                            "dte_open": _sanitize_float(leg.dte_open),
                            "pnl": _sanitize_float(leg.pnl),
                            "trigger": leg.trigger,
                        }
                        for leg in chain.legs
                    ],
                })
            
            equity_curve_json = [
                {
                    "time": pt.time.isoformat(),
                    "equity": _sanitize_float(pt.equity),
                    "hodl_equity": _sanitize_float(pt.hodl_equity),
                }
                for pt in self._status.equity_curve
            ]
            
            status_dict = {
                "running": self._status.running,
                "paused": self._status.paused,
                "started_at": self._status.started_at.isoformat() if self._status.started_at else None,
                "finished_at": self._status.finished_at.isoformat() if self._status.finished_at else None,
                "progress_pct": _sanitize_float(self._status.progress_pct),
                "current_time": self._status.current_time.isoformat() if self._status.current_time else None,
                "decisions_processed": self._status.decisions_processed,
                "total_decisions": self._status.total_decisions,
                "exit_style": self._status.exit_style,
                "current_phase": self._status.current_phase,
                "underlying": self._status.underlying,
                "start_date": self._status.start_date.isoformat() if self._status.start_date else None,
                "end_date": self._status.end_date.isoformat() if self._status.end_date else None,
                "metrics": _sanitize_dict(self._status.metrics) if self._status.metrics else {},
                "recent_steps": [
                    {
                        "time": step.time.isoformat(),
                        "candidates": step.candidates,
                        "best_score": _sanitize_float(step.best_score),
                        "traded": step.traded,
                        "exit_style": step.exit_style,
                    }
                    for step in self._status.recent_steps
                ],
                "recent_chains": chains_json,
                "equity_curve": equity_curve_json,
                "error": self._status.error,
            }
            return status_dict
    
    def _append_chain_summary(self, trade: Any, exit_style: str) -> None:
        """Extract chain data from trade and append to recent_chains."""
        chain = getattr(trade, "chain", None)
        if not chain:
            return

        legs = []
        for leg in getattr(chain, "legs", []):
            legs.append(BacktestLegSummary(
                index=leg.index,
                open_time=leg.open_time,
                close_time=leg.close_time,
                strike=float(leg.strike),
                dte_open=float(leg.dte_open),
                pnl=float(leg.pnl),
                trigger=leg.trigger,
            ))

        num_legs = len(legs)
        num_rolls = max(0, num_legs - 1)

        summary = BacktestChainSummary(
            decision_time=chain.decision_time,
            underlying=getattr(chain, "underlying", self._status.underlying or ""),
            num_legs=num_legs,
            num_rolls=num_rolls,
            total_pnl=float(chain.total_pnl),
            max_drawdown_pct=float(chain.max_drawdown_pct),
            exit_style=exit_style,
            legs=legs,
        )

        with self._lock:
            self._status.recent_chains.append(summary)
            self._status.recent_chains = self._status.recent_chains[-MAX_RECENT_CHAINS:]

    def stop(self) -> None:
        with self._lock:
            self._cancel_requested = True
            self._pause_event.set()

    def pause(self) -> None:
        with self._lock:
            if self._status.running and not self._status.paused:
                self._pause_event.clear()
                self._status.paused = True

    def resume(self) -> None:
        with self._lock:
            if self._status.running and self._status.paused:
                self._pause_event.set()
                self._status.paused = False

    def _set_error(self, msg: str) -> None:
        with self._lock:
            self._status.error = msg
            self._status.running = False
            self._status.paused = False
            self._status.finished_at = datetime.utcnow()

    def _update_status_step(
        self,
        processed: int,
        total: int,
        current_time: datetime,
        steps_buffer: List[BacktestProgressStep],
        progress_pct: float,
        current_phase: Optional[str] = None,
    ) -> None:
        with self._lock:
            self._status.decisions_processed = processed
            self._status.total_decisions = total
            self._status.current_time = current_time
            self._status.progress_pct = progress_pct
            self._status.recent_steps = steps_buffer[-50:]
            if current_phase:
                self._status.current_phase = current_phase
    
    def _export_training_data_if_enabled(
        self,
        underlying: str,
        start_date: datetime,
        end_date: datetime,
        exit_style: str,
        examples: List[Any],
    ) -> None:
        """Export training data if SAVE_TRAINING_DATA is enabled."""
        try:
            from src.config import settings
            if not settings.save_training_data:
                return
            
            from pathlib import Path
            from .training_dataset import export_to_csv, export_to_jsonl, compute_dataset_stats
            
            if not examples:
                print(f"[TrainingExport] No training examples generated for {underlying}")
                return
            
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            base_name = f"training_dataset_{underlying}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}_{exit_style}_{timestamp}"
            
            data_dir = Path(settings.training_data_dir)
            data_dir.mkdir(parents=True, exist_ok=True)
            
            csv_path = data_dir / f"{base_name}.csv"
            jsonl_path = data_dir / f"{base_name}.jsonl"
            
            export_to_csv(examples, csv_path)
            export_to_jsonl(examples, jsonl_path)
            
            stats = compute_dataset_stats(examples)
            print(f"[TrainingExport] Saved {stats['total_examples']} examples to {csv_path} and {jsonl_path}")
            print(f"[TrainingExport] Stats: {stats}")
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[TrainingExport] Error exporting training data: {e}")

    def start(
        self,
        underlying: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str,
        decision_interval_hours: int,
        exit_style: str,
        target_dte: int = 7,
        target_delta: float = 0.25,
        min_dte: int = 1,
        max_dte: int = 30,
        delta_min: float = 0.10,
        delta_max: float = 0.45,
        margin_type: MarginType = "inverse",
        settlement_ccy: SettlementCcy = "ANY",
    ) -> bool:
        with self._lock:
            if self._status.running:
                return False
            self._cancel_requested = False
            self._pause_event.set()
            self._status = BacktestStatus(
                running=True,
                paused=False,
                started_at=datetime.utcnow(),
                finished_at=None,
                progress_pct=0.0,
                current_time=None,
                decisions_processed=0,
                total_decisions=0,
                exit_style=exit_style,
                current_phase=None,
                underlying=underlying,
                start_date=start_date,
                end_date=end_date,
                metrics={},
                recent_steps=[],
                error=None,
            )

        def worker() -> None:
            try:
                from src.backtest.types import CallSimulationConfig
                from src.backtest.deribit_data_source import DeribitDataSource
                from src.backtest.covered_call_simulator import CoveredCallSimulator
                from src.backtest.state_builder import build_historical_state
                from datetime import timedelta
                from typing import cast
                from src.backtest.data_source import Timeframe

                tf: Timeframe = cast(Timeframe, timeframe)
                
                hours_per_bar = {"1m": 1/60, "5m": 5/60, "15m": 0.25, "1h": 1, "4h": 4, "1d": 24}
                bar_duration_hours = hours_per_bar.get(timeframe, 1)
                
                interval_bars = max(1, int(decision_interval_hours / bar_duration_hours))
                
                config = CallSimulationConfig(
                    underlying=underlying,
                    start=start_date,
                    end=end_date,
                    timeframe=tf,
                    decision_interval_bars=interval_bars,
                    initial_spot_position=1.0,
                    contract_size=1.0,
                    fee_rate=0.0005,
                    target_dte=target_dte,
                    dte_tolerance=3,
                    target_delta=target_delta,
                    delta_tolerance=0.15,
                    min_dte=min_dte,
                    max_dte=max_dte,
                    delta_min=delta_min,
                    delta_max=delta_max,
                    option_margin_type=margin_type,
                    option_settlement_ccy=settlement_ccy,
                )

                ds = DeribitDataSource()
                sim = CoveredCallSimulator(data_source=ds, config=config)

                decision_times: List[datetime] = []
                current = start_date
                while current <= end_date:
                    decision_times.append(current)
                    current = current + timedelta(hours=decision_interval_hours)
                
                if not decision_times:
                    decision_times = [start_date]

                styles_to_run: List[str] = []
                if exit_style == "both":
                    styles_to_run = ["hold_to_expiry", "tp_and_roll"]
                else:
                    styles_to_run = [exit_style]

                total_phases = len(styles_to_run)
                all_metrics: Dict[str, Any] = {}

                with self._lock:
                    total_all = len(decision_times) * total_phases
                    self._status.total_decisions = total_all

                global_step_index = 0

                all_training_examples: List[Any] = []
                spot_at_times: Dict[datetime, float] = {}
                all_equity_curves: Dict[str, List[EquityCurvePoint]] = {}
                position_size = config.initial_spot_position
                
                for phase_idx, current_exit_style in enumerate(styles_to_run):
                    steps_buffer: List[BacktestProgressStep] = []
                    trades: List[Any] = []
                    cumulative_pnl = 0.0
                    cumulative_pnl_vs_hodl = 0.0

                    with self._lock:
                        self._status.current_phase = current_exit_style

                    for idx, t in enumerate(decision_times):
                        self._pause_event.wait()
                        
                        with self._lock:
                            if self._cancel_requested:
                                self._status.running = False
                                self._status.paused = False
                                self._status.finished_at = datetime.utcnow()
                                self._status.error = "Cancelled by user"
                                ds.close()
                                return

                        try:
                            state = build_historical_state(ds, config, t)
                        except Exception:
                            state = {}

                        spot = state.get("spot")
                        options = state.get("candidate_options") or []
                        
                        if spot and spot > 0:
                            spot_at_times[t] = float(spot)
                        
                        if spot is None or spot <= 0 or not options:
                            step = BacktestProgressStep(
                                time=t,
                                candidates=len(options),
                                best_score=0.0,
                                traded=False,
                                exit_style=current_exit_style,
                            )
                            steps_buffer.append(step)
                            global_step_index += 1
                            progress = global_step_index / (len(decision_times) * total_phases)
                            self._update_status_step(global_step_index, len(decision_times) * total_phases, t, steps_buffer, progress, current_exit_style)
                            continue

                        scored = []
                        for opt in options:
                            try:
                                feats = sim._extract_candidate_features(state, opt)
                                s = sim._score_candidate(feats)
                                scored.append((s, opt))
                            except Exception:
                                continue

                        if not scored:
                            step = BacktestProgressStep(
                                time=t,
                                candidates=len(options),
                                best_score=0.0,
                                traded=False,
                                exit_style=current_exit_style,
                            )
                            steps_buffer.append(step)
                            global_step_index += 1
                            progress = global_step_index / (len(decision_times) * total_phases)
                            self._update_status_step(global_step_index, len(decision_times) * total_phases, t, steps_buffer, progress, current_exit_style)
                            continue

                        scored.sort(key=lambda x: x[0], reverse=True)
                        best_score, best_opt = scored[0]

                        traded = False
                        trade = None
                        min_score = getattr(config, 'min_score_to_trade', 3.0)
                        if best_score >= min_score:
                            try:
                                if current_exit_style == "hold_to_expiry":
                                    trade = sim._simulate_call_hold_to_expiry(t, best_opt)
                                else:
                                    trade = sim._simulate_call_tp_and_roll(t, best_opt)
                            except Exception:
                                trade = None

                        from .types import TrainingExample
                        if trade is not None:
                            trades.append(trade)
                            cumulative_pnl += trade.pnl
                            cumulative_pnl_vs_hodl += trade.pnl_vs_hodl
                            traded = True
                            self._append_chain_summary(trade, current_exit_style)
                            
                            all_training_examples.append(TrainingExample(
                                decision_time=t,
                                underlying=underlying,
                                spot=spot,
                                action="SELL_CALL",
                                reward=trade.pnl,
                                extra={
                                    "instrument": best_opt.instrument_name,
                                    "delta": best_opt.delta,
                                    "strike": best_opt.strike,
                                    "dte": (best_opt.expiry - t).days,
                                    "score": best_score,
                                    "exit_style": current_exit_style,
                                    "pnl_vs_hodl": trade.pnl_vs_hodl,
                                    "max_drawdown_pct": trade.max_drawdown_pct,
                                },
                            ))
                        else:
                            all_training_examples.append(TrainingExample(
                                decision_time=t,
                                underlying=underlying,
                                spot=spot if spot else 0.0,
                                action="DO_NOTHING",
                                reward=0.0,
                                extra={
                                    "best_score": best_score,
                                    "candidates": len(options),
                                    "reason": "score_below_threshold" if best_score < min_score else "trade_simulation_failed",
                                },
                            ))

                        step = BacktestProgressStep(
                            time=t,
                            candidates=len(options),
                            best_score=round(best_score, 2),
                            traded=traded,
                            exit_style=current_exit_style,
                        )
                        steps_buffer.append(step)
                        global_step_index += 1
                        progress = global_step_index / (len(decision_times) * total_phases)
                        self._update_status_step(global_step_index, len(decision_times) * total_phases, t, steps_buffer, progress, current_exit_style)

                    equity_curve = _compute_equity_curve(
                        trades=trades,
                        spot_at_times=spot_at_times,
                        position_size=position_size,
                        start_time=start_date,
                        end_time=end_date,
                    )
                    all_equity_curves[current_exit_style] = equity_curve
                    
                    phase_metrics = _compute_enhanced_metrics(
                        trades=trades,
                        equity_curve=equity_curve,
                        position_size=position_size,
                    )

                    if exit_style == "both":
                        all_metrics[current_exit_style] = phase_metrics
                    else:
                        all_metrics = phase_metrics

                ds.close()
                
                self._export_training_data_if_enabled(
                    underlying=underlying,
                    start_date=start_date,
                    end_date=end_date,
                    exit_style=exit_style,
                    examples=all_training_examples,
                )

                primary_equity_curve: List[EquityCurvePoint] = []
                if exit_style == "both":
                    primary_equity_curve = all_equity_curves.get("tp_and_roll", [])
                else:
                    primary_equity_curve = all_equity_curves.get(exit_style, [])

                with self._lock:
                    self._status.metrics = all_metrics
                    self._status.equity_curve = primary_equity_curve
                    self._status.progress_pct = 1.0
                    self._status.running = False
                    self._status.paused = False
                    self._status.finished_at = datetime.utcnow()
                    self._status.current_phase = None

            except Exception as e:
                import traceback
                traceback.print_exc()
                self._set_error(str(e))

        self._thread = Thread(target=worker, daemon=True)
        self._thread.start()
        return True


backtest_manager = BacktestManager()
