"""Shared volatility metrics calculations.

This module centralizes calculations related to implied and realized volatility
to ensure consistency across the live agent, backtests, and training data.
"""


def compute_ivrv_ratio(
    iv: float | None,
    realized_vol: float | None,
    default: float = 1.0,
) -> float:
    """Compute the IV/RV ratio (implied volatility / realized volatility).

    The IV/RV ratio measures how much option implied volatility exceeds
    realized (historical) volatility. A ratio > 1.0 means options are
    "expensive" relative to recent actual price movement, which is
    typically favorable for premium sellers.

    Args:
        iv: Implied volatility (annualized, e.g. 0.70 for 70%).
        realized_vol: Realized volatility (annualized, e.g. 0.50 for 50%).
        default: Value to return when calculation is not possible (default 1.0).

    Returns:
        The IV/RV ratio as a float, or the default value if either input
        is None, zero, or negative.

    Examples:
        >>> compute_ivrv_ratio(0.70, 0.50)
        1.4
        >>> compute_ivrv_ratio(None, 0.50)
        1.0
        >>> compute_ivrv_ratio(0.70, 0.0)
        1.0
    """
    if iv is None or realized_vol is None:
        return default
    if realized_vol <= 0:
        return default
    if iv <= 0:
        return default
    return iv / realized_vol
