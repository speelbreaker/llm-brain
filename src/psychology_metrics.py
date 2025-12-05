"""
Psychology metrics module for strategy evaluation.
Computes expectancy, Monte Carlo variance, and psychology-friendliness scores.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Sequence


@dataclass
class TradeResult:
    """
    Generic trade or chain result.
    Treat one "trade" as one completed options chain if that fits better.
    """
    timestamp: datetime
    pnl_pct: float
    is_win: bool
    underlying: str
    chain_id: Optional[str] = None


@dataclass
class StrategyStats:
    """Computed statistics for a trading strategy."""
    n_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    expectancy_per_trade: float
    max_consec_losses: int
    max_consec_wins: int
    max_drawdown_pct: float
    years_covered: float


@dataclass
class VarianceProfile:
    """Monte Carlo simulation results showing variance characteristics."""
    sims: int
    trades_per_sim: int
    median_final_return: float
    worst_final_return: float
    best_final_return: float
    median_max_dd: float
    worst_max_dd: float
    median_max_loss_streak: int
    worst_max_loss_streak: int


@dataclass
class PsychologyScore:
    """Psychology-friendliness score for a strategy."""
    score: float
    components: Dict[str, float]


def compute_strategy_stats(
    trades: Sequence[TradeResult],
    initial_equity: float = 1.0,
) -> StrategyStats:
    """
    Compute strategy statistics from a sequence of trade results.
    
    Args:
        trades: Sequence of TradeResult objects
        initial_equity: Starting equity (default 1.0 for percentage-based)
    
    Returns:
        StrategyStats with computed metrics
    
    Raises:
        ValueError: If no trades provided
    """
    if not trades:
        raise ValueError("No trades provided")

    n = len(trades)
    wins = [t.pnl_pct for t in trades if t.pnl_pct > 0]
    losses = [t.pnl_pct for t in trades if t.pnl_pct < 0]

    win_rate = len(wins) / n if n > 0 else 0.0
    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = sum(losses) / len(losses) if losses else 0.0

    expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss

    equity = initial_equity
    peak = initial_equity
    max_dd = 0.0
    max_consec_losses = 0
    max_consec_wins = 0
    cur_losses = 0
    cur_wins = 0

    for t in trades:
        equity *= (1 + t.pnl_pct)
        peak = max(peak, equity)
        dd = (equity - peak) / peak
        max_dd = min(max_dd, dd)

        if t.pnl_pct > 0:
            cur_wins += 1
            cur_losses = 0
        elif t.pnl_pct < 0:
            cur_losses += 1
            cur_wins = 0

        max_consec_losses = max(max_consec_losses, cur_losses)
        max_consec_wins = max(max_consec_wins, cur_wins)

    sorted_trades = sorted(trades, key=lambda x: x.timestamp)
    span_days = (sorted_trades[-1].timestamp - sorted_trades[0].timestamp).days or 1
    years_covered = span_days / 365.25

    return StrategyStats(
        n_trades=n,
        win_rate=win_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        expectancy_per_trade=expectancy,
        max_consec_losses=max_consec_losses,
        max_consec_wins=max_consec_wins,
        max_drawdown_pct=max_dd,
        years_covered=years_covered,
    )


def _simulate_equity(
    returns: Sequence[float],
    initial_equity: float = 1.0,
) -> tuple[float, float, int]:
    """
    Simulate equity curve from a sequence of returns.
    
    Returns:
        Tuple of (final_equity, max_drawdown, max_loss_streak)
    """
    equity = initial_equity
    peak = initial_equity
    max_dd = 0.0
    max_loss_streak = 0
    cur_loss_streak = 0

    for r in returns:
        equity *= (1 + r)
        peak = max(peak, equity)
        dd = (equity - peak) / peak
        max_dd = min(max_dd, dd)

        if r < 0:
            cur_loss_streak += 1
        else:
            cur_loss_streak = 0

        max_loss_streak = max(max_loss_streak, cur_loss_streak)

    return equity, max_dd, max_loss_streak


def run_monte_carlo(
    trades: Sequence[TradeResult],
    sims: int = 1000,
    trades_per_sim: Optional[int] = None,
    bootstrap: bool = True,
    seed: Optional[int] = None,
) -> VarianceProfile:
    """
    Run Monte Carlo simulation to assess strategy variance.
    
    Args:
        trades: Historical trade results
        sims: Number of simulations to run
        trades_per_sim: Number of trades per simulation (default: same as input)
        bootstrap: If True, sample with replacement; if False, shuffle
        seed: Random seed for reproducibility
    
    Returns:
        VarianceProfile with simulation results
    
    Raises:
        ValueError: If no trades provided
    """
    if not trades:
        raise ValueError("No trades provided")

    rng = random.Random(seed)
    returns = [t.pnl_pct for t in trades]
    n = len(returns)
    trades_per_sim = trades_per_sim or n

    final_eqs: List[float] = []
    max_dds: List[float] = []
    max_streaks: List[int] = []

    for _ in range(sims):
        if bootstrap:
            path = [rng.choice(returns) for _ in range(trades_per_sim)]
        else:
            path = returns.copy()
            rng.shuffle(path)
            if trades_per_sim < n:
                path = path[:trades_per_sim]

        final_eq, max_dd, max_streak = _simulate_equity(path)
        final_eqs.append(final_eq)
        max_dds.append(max_dd)
        max_streaks.append(max_streak)

    def _percentile(data: List[float], p: float) -> float:
        if not data:
            return 0.0
        data = sorted(data)
        k = (len(data) - 1) * p
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return data[int(k)]
        d0 = data[f] * (c - k)
        d1 = data[c] * (k - f)
        return d0 + d1

    median_final = _percentile(final_eqs, 0.5) - 1.0
    worst_final = min(final_eqs) - 1.0
    best_final = max(final_eqs) - 1.0

    median_dd = _percentile(max_dds, 0.5)
    worst_dd = min(max_dds)

    median_streak = int(round(_percentile([float(x) for x in max_streaks], 0.5)))
    worst_streak = max(max_streaks)

    return VarianceProfile(
        sims=sims,
        trades_per_sim=trades_per_sim,
        median_final_return=median_final,
        worst_final_return=worst_final,
        best_final_return=best_final,
        median_max_dd=median_dd,
        worst_max_dd=worst_dd,
        median_max_loss_streak=median_streak,
        worst_max_loss_streak=worst_streak,
    )


def compute_psychology_score(
    stats: StrategyStats,
    variance: VarianceProfile,
    target_expectancy_per_trade: float = 0.001,
) -> PsychologyScore:
    """
    Compute a psychology-friendliness score for a strategy.
    
    This measures how "smooth" and "human-tolerable" a strategy is:
    - High win rate is psychologically easier
    - Low loss streaks reduce emotional stress
    - Low drawdowns prevent panic selling
    
    Args:
        stats: Strategy statistics
        variance: Monte Carlo variance profile
        target_expectancy_per_trade: Target expectancy for scoring (default 0.1%)
    
    Returns:
        PsychologyScore with overall score (0-100 range) and components
    """
    win_rate_score = min(max((stats.win_rate - 0.4) / 0.4, 0.0), 1.0)
    exp_score = min(max(stats.expectancy_per_trade / target_expectancy_per_trade, 0.0), 1.5)

    streak_penalty = min(variance.worst_max_loss_streak / 10.0, 2.0)
    dd_penalty = min(abs(variance.worst_max_dd) / 0.3, 2.0)

    raw_score = (
        0.4 * win_rate_score +
        0.3 * exp_score -
        0.2 * streak_penalty -
        0.1 * dd_penalty
    )

    final_score = max(raw_score, -2.0) + 2.0
    final_score *= 25.0

    components = {
        "win_rate_score": win_rate_score,
        "expectancy_score": exp_score,
        "streak_penalty": streak_penalty,
        "dd_penalty": dd_penalty,
    }
    return PsychologyScore(score=final_score, components=components)
