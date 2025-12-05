"""
Backtest Manager for running backtests with live progress tracking.
Only one active backtest at a time.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from threading import Thread, Lock
from typing import Any, Dict, List, Literal, Optional

ExitStyle = Literal["hold_to_expiry", "tp_and_roll"]


@dataclass
class BacktestProgressStep:
    time: datetime
    candidates: int
    best_score: float
    traded: bool
    exit_style: ExitStyle


@dataclass
class BacktestStatus:
    running: bool = False
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    progress_pct: float = 0.0
    current_time: Optional[datetime] = None
    decisions_processed: int = 0
    total_decisions: int = 0
    exit_style: Optional[ExitStyle] = None
    underlying: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    recent_steps: List[BacktestProgressStep] = field(default_factory=list)
    error: Optional[str] = None


class BacktestManager:
    def __init__(self) -> None:
        self._lock = Lock()
        self._thread: Optional[Thread] = None
        self._status = BacktestStatus()
        self._cancel_requested = False

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            status_dict = {
                "running": self._status.running,
                "started_at": self._status.started_at.isoformat() if self._status.started_at else None,
                "finished_at": self._status.finished_at.isoformat() if self._status.finished_at else None,
                "progress_pct": self._status.progress_pct,
                "current_time": self._status.current_time.isoformat() if self._status.current_time else None,
                "decisions_processed": self._status.decisions_processed,
                "total_decisions": self._status.total_decisions,
                "exit_style": self._status.exit_style,
                "underlying": self._status.underlying,
                "start_date": self._status.start_date.isoformat() if self._status.start_date else None,
                "end_date": self._status.end_date.isoformat() if self._status.end_date else None,
                "metrics": self._status.metrics,
                "recent_steps": [
                    {
                        "time": step.time.isoformat(),
                        "candidates": step.candidates,
                        "best_score": step.best_score,
                        "traded": step.traded,
                        "exit_style": step.exit_style,
                    }
                    for step in self._status.recent_steps
                ],
                "error": self._status.error,
            }
            return status_dict

    def stop(self) -> None:
        with self._lock:
            self._cancel_requested = True

    def _set_error(self, msg: str) -> None:
        with self._lock:
            self._status.error = msg
            self._status.running = False
            self._status.finished_at = datetime.utcnow()

    def _update_status_step(
        self,
        processed: int,
        total: int,
        current_time: datetime,
        steps_buffer: List[BacktestProgressStep],
        cumulative_pnl: float,
    ) -> None:
        with self._lock:
            self._status.decisions_processed = processed
            self._status.total_decisions = total
            self._status.current_time = current_time
            self._status.progress_pct = processed / total if total else 0.0
            self._status.recent_steps = steps_buffer[-50:]

    def start(
        self,
        underlying: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str,
        decision_interval_hours: int,
        exit_style: ExitStyle,
        target_dte: int = 7,
        target_delta: float = 0.25,
    ) -> bool:
        with self._lock:
            if self._status.running:
                return False
            self._cancel_requested = False
            self._status = BacktestStatus(
                running=True,
                started_at=datetime.utcnow(),
                finished_at=None,
                progress_pct=0.0,
                current_time=None,
                decisions_processed=0,
                total_decisions=0,
                exit_style=exit_style,
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
                from src.backtest.market_context_backtest import compute_market_context_from_ds
                from datetime import timedelta
                from typing import cast
                from src.backtest.data_source import Timeframe

                tf: Timeframe = cast(Timeframe, timeframe)
                
                config = CallSimulationConfig(
                    underlying=underlying,
                    start=start_date,
                    end=end_date,
                    timeframe=tf,
                    decision_interval_bars=1,
                    initial_spot_position=1.0,
                    contract_size=1.0,
                    fee_rate=0.0005,
                    target_dte=target_dte,
                    dte_tolerance=3,
                    target_delta=target_delta,
                    delta_tolerance=0.15,
                )

                ds = DeribitDataSource()
                sim = CoveredCallSimulator(data_source=ds, config=config)

                hours_map = {"1m": 1/60, "5m": 5/60, "15m": 0.25, "1h": 1, "4h": 4, "1d": 24}
                bar_hours = hours_map.get(timeframe, 1)
                interval_hours = decision_interval_hours * bar_hours

                decision_times: List[datetime] = []
                current = start_date
                while current <= end_date:
                    decision_times.append(current)
                    current = current + timedelta(hours=interval_hours)

                with self._lock:
                    self._status.total_decisions = len(decision_times)

                steps_buffer: List[BacktestProgressStep] = []
                trades: List[Any] = []
                cumulative_pnl = 0.0
                cumulative_pnl_vs_hodl = 0.0

                for idx, t in enumerate(decision_times):
                    with self._lock:
                        if self._cancel_requested:
                            break

                    try:
                        state = build_historical_state(ds, config, t)
                    except Exception:
                        state = {}

                    spot = state.get("spot")
                    options = state.get("candidate_options") or []
                    
                    if spot is None or spot <= 0 or not options:
                        step = BacktestProgressStep(
                            time=t,
                            candidates=len(options),
                            best_score=0.0,
                            traded=False,
                            exit_style=exit_style,
                        )
                        steps_buffer.append(step)
                        self._update_status_step(idx + 1, len(decision_times), t, steps_buffer, cumulative_pnl)
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
                            exit_style=exit_style,
                        )
                        steps_buffer.append(step)
                        self._update_status_step(idx + 1, len(decision_times), t, steps_buffer, cumulative_pnl)
                        continue

                    scored.sort(key=lambda x: x[0], reverse=True)
                    best_score, best_opt = scored[0]

                    traded = False
                    trade = None
                    min_score = getattr(config, 'min_score_to_trade', 3.0)
                    if best_score >= min_score:
                        try:
                            if exit_style == "hold_to_expiry":
                                trade = sim._simulate_call_hold_to_expiry(t, best_opt)
                            else:
                                trade = sim._simulate_call_tp_and_roll(t, best_opt)
                        except Exception:
                            trade = None

                    if trade is not None:
                        trades.append(trade)
                        cumulative_pnl += trade.pnl
                        cumulative_pnl_vs_hodl += trade.pnl_vs_hodl
                        traded = True

                    step = BacktestProgressStep(
                        time=t,
                        candidates=len(options),
                        best_score=round(best_score, 2),
                        traded=traded,
                        exit_style=exit_style,
                    )
                    steps_buffer.append(step)

                    self._update_status_step(idx + 1, len(decision_times), t, steps_buffer, cumulative_pnl)

                ds.close()

                win_count = sum(1 for t in trades if t.pnl > 0)
                metrics = {
                    "num_trades": len(trades),
                    "final_pnl": round(cumulative_pnl, 4),
                    "final_pnl_vs_hodl": round(cumulative_pnl_vs_hodl, 4),
                    "avg_pnl": round(cumulative_pnl / len(trades), 4) if trades else 0,
                    "win_rate": round(win_count / len(trades) * 100, 1) if trades else 0,
                }

                with self._lock:
                    self._status.metrics = metrics
                    self._status.progress_pct = 1.0
                    self._status.running = False
                    self._status.finished_at = datetime.utcnow()

            except Exception as e:
                import traceback
                traceback.print_exc()
                self._set_error(str(e))

        self._thread = Thread(target=worker, daemon=True)
        self._thread.start()
        return True


backtest_manager = BacktestManager()
