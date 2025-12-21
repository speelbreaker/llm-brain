"""Unit sanity helpers for backtests.

Backtests and fidelity runs assume linear USDC settlement, so option premiums
must be expressed in USD/USDC (not underlying units like BTC).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class UnitSanityConfig:
    """Heuristics for detecting unit mistakes."""

    # Premiums should not exceed an extreme fraction of spot for our typical
    # filtered calls (e.g., ~0.10-0.40 delta). Keep this loose to avoid false
    # positives while still catching BTC-vs-USD mistakes.
    max_premium_fraction_of_spot: float = 0.75
    min_spot_for_checks: float = 100.0


def premium_underlying_to_usd(premium_underlying: float, spot_usd: float) -> float:
    """Convert a premium quoted in underlying units (e.g., BTC) into USD."""
    if premium_underlying is None:
        return 0.0
    if spot_usd is None or spot_usd <= 0:
        return 0.0
    return float(premium_underlying) * float(spot_usd)


def assert_premium_usd_sane(
    premium_usd: float,
    spot_usd: float,
    *,
    cfg: UnitSanityConfig | None = None,
    context: str = "",
) -> None:
    """Raise ValueError if a premium looks like it's in the wrong units."""
    if premium_usd is None:
        return
    if spot_usd is None or spot_usd <= 0:
        return

    cfg = cfg or UnitSanityConfig()
    if spot_usd < cfg.min_spot_for_checks:
        return

    premium = float(premium_usd)
    spot = float(spot_usd)

    if premium < 0:
        raise ValueError(f"Negative option premium (usd) {premium:.6f}. {context}".strip())

    if premium > spot * cfg.max_premium_fraction_of_spot:
        raise ValueError(
            f"Option premium looks too large for USD units: premium={premium:.4f}, spot={spot:.2f}. {context}".strip()
        )
