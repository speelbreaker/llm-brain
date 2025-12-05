"""
Strategy approval module for evaluating strategy readiness.
Determines if a strategy is ready for research, pilot, or production deployment.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.psychology_metrics import PsychologyScore, StrategyStats, VarianceProfile


@dataclass
class ApprovalConfig:
    """Configuration for strategy approval thresholds."""
    min_trades: int = 200
    min_years: float = 3.0
    require_bull_bear_chop: bool = True

    max_worst_dd_research: float = 0.50
    max_worst_dd_pilot: float = 0.35
    max_worst_dd_prod: float = 0.25

    max_loss_streak_pilot: int = 8
    max_loss_streak_prod: int = 6

    min_psych_score_pilot: float = 40.0
    min_psych_score_prod: float = 60.0

    require_param_sensitivity_ok: bool = True
    require_walk_forward_ok: bool = True
    require_stress_test_ok: bool = True


@dataclass
class RegimeCoverage:
    """Tracks which market regimes the strategy has been tested in."""
    bull: bool
    bear: bool
    chop: bool


@dataclass
class ApprovalResult:
    """Result of strategy approval evaluation."""
    stage: str
    reason: str
    stats: StrategyStats
    variance: VarianceProfile
    psych: PsychologyScore


def evaluate_strategy(
    stats: StrategyStats,
    variance: VarianceProfile,
    psych: PsychologyScore,
    regimes: RegimeCoverage,
    param_sensitivity_ok: bool,
    walk_forward_ok: bool,
    stress_test_ok: bool,
    cfg: Optional[ApprovalConfig] = None,
) -> ApprovalResult:
    """
    Evaluate a strategy's readiness for different deployment stages.
    
    Stages:
    - RESEARCH: Strategy needs more data/testing
    - PILOT: Ready for limited live testing
    - PRODUCTION: Ready for full deployment
    
    Args:
        stats: Strategy statistics
        variance: Monte Carlo variance profile
        psych: Psychology score
        regimes: Market regime coverage
        param_sensitivity_ok: Whether parameter sensitivity is acceptable
        walk_forward_ok: Whether walk-forward testing passed
        stress_test_ok: Whether stress testing passed
        cfg: Approval configuration thresholds
    
    Returns:
        ApprovalResult with stage, reason, and metrics
    """
    cfg = cfg or ApprovalConfig()

    if stats.n_trades < cfg.min_trades:
        return ApprovalResult(
            stage="RESEARCH",
            reason=f"Insufficient sample size ({stats.n_trades} < {cfg.min_trades})",
            stats=stats,
            variance=variance,
            psych=psych,
        )

    if stats.years_covered < cfg.min_years:
        return ApprovalResult(
            stage="RESEARCH",
            reason=f"Insufficient history ({stats.years_covered:.1f}y < {cfg.min_years:.1f}y)",
            stats=stats,
            variance=variance,
            psych=psych,
        )

    if cfg.require_bull_bear_chop and not (regimes.bull and regimes.bear and regimes.chop):
        missing = []
        if not regimes.bull:
            missing.append("bull")
        if not regimes.bear:
            missing.append("bear")
        if not regimes.chop:
            missing.append("chop")
        return ApprovalResult(
            stage="RESEARCH",
            reason=f"Missing regime coverage: {', '.join(missing)}",
            stats=stats,
            variance=variance,
            psych=psych,
        )

    if abs(variance.worst_max_dd) > cfg.max_worst_dd_research:
        return ApprovalResult(
            stage="RESEARCH",
            reason=f"Worst drawdown too severe ({variance.worst_max_dd:.1%} > {cfg.max_worst_dd_research:.1%})",
            stats=stats,
            variance=variance,
            psych=psych,
        )

    if abs(variance.worst_max_dd) > cfg.max_worst_dd_pilot:
        return ApprovalResult(
            stage="RESEARCH",
            reason=f"Drawdown too high for pilot ({variance.worst_max_dd:.1%} > {cfg.max_worst_dd_pilot:.1%})",
            stats=stats,
            variance=variance,
            psych=psych,
        )

    if variance.worst_max_loss_streak > cfg.max_loss_streak_pilot:
        return ApprovalResult(
            stage="RESEARCH",
            reason=f"Loss streak too long for pilot ({variance.worst_max_loss_streak} > {cfg.max_loss_streak_pilot})",
            stats=stats,
            variance=variance,
            psych=psych,
        )

    if psych.score < cfg.min_psych_score_pilot:
        return ApprovalResult(
            stage="RESEARCH",
            reason=f"Psychology score too low for pilot ({psych.score:.1f} < {cfg.min_psych_score_pilot:.1f})",
            stats=stats,
            variance=variance,
            psych=psych,
        )

    if abs(variance.worst_max_dd) > cfg.max_worst_dd_prod:
        return ApprovalResult(
            stage="PILOT",
            reason=f"Drawdown acceptable for pilot but not production ({variance.worst_max_dd:.1%})",
            stats=stats,
            variance=variance,
            psych=psych,
        )

    if variance.worst_max_loss_streak > cfg.max_loss_streak_prod:
        return ApprovalResult(
            stage="PILOT",
            reason=f"Loss streak acceptable for pilot but not production ({variance.worst_max_loss_streak})",
            stats=stats,
            variance=variance,
            psych=psych,
        )

    if psych.score < cfg.min_psych_score_prod:
        return ApprovalResult(
            stage="PILOT",
            reason=f"Psychology score acceptable for pilot but not production ({psych.score:.1f})",
            stats=stats,
            variance=variance,
            psych=psych,
        )

    if cfg.require_param_sensitivity_ok and not param_sensitivity_ok:
        return ApprovalResult(
            stage="PILOT",
            reason="Parameter sensitivity not validated",
            stats=stats,
            variance=variance,
            psych=psych,
        )

    if cfg.require_walk_forward_ok and not walk_forward_ok:
        return ApprovalResult(
            stage="PILOT",
            reason="Walk-forward testing not validated",
            stats=stats,
            variance=variance,
            psych=psych,
        )

    if cfg.require_stress_test_ok and not stress_test_ok:
        return ApprovalResult(
            stage="PILOT",
            reason="Stress testing not validated",
            stats=stats,
            variance=variance,
            psych=psych,
        )

    return ApprovalResult(
        stage="PRODUCTION",
        reason="All criteria met for production deployment",
        stats=stats,
        variance=variance,
        psych=psych,
    )


def format_approval_summary(result: ApprovalResult) -> str:
    """Format an approval result as a human-readable summary."""
    lines = [
        f"Strategy Approval: {result.stage}",
        f"Reason: {result.reason}",
        "",
        "Stats:",
        f"  Trades: {result.stats.n_trades}",
        f"  Win Rate: {result.stats.win_rate:.1%}",
        f"  Expectancy: {result.stats.expectancy_per_trade:.3%} per trade",
        f"  Max DD: {result.stats.max_drawdown_pct:.1%}",
        f"  Max Loss Streak: {result.stats.max_consec_losses}",
        "",
        "Variance Profile:",
        f"  Worst DD (MC): {result.variance.worst_max_dd:.1%}",
        f"  Worst Loss Streak (MC): {result.variance.worst_max_loss_streak}",
        "",
        f"Psychology Score: {result.psych.score:.1f}/100",
    ]
    return "\n".join(lines)
