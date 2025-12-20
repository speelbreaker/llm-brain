from __future__ import annotations

from typing import Any, Dict

from .canonical_strategies import canonical_strategies


def fidelity_spec() -> Dict[str, Any]:
    """Return a stable description of the fidelity suite contract (MVP)."""
    strategies = [s.name for s in canonical_strategies()]

    components = {
        "underlying_returns": {
            "weight": 0.20,
            "metrics": {
                "tail_quantile_diff": {"tolerance": 0.02, "weight": 0.6},
                "rv_level_diff": {"tolerance": 0.10, "weight": 0.4},
            },
        },
        "iv_surface_level": {
            "weight": 0.30,
            "metrics": {
                "iv_bucket_mae": {"tolerance": 0.05, "weight": 1.0},
            },
        },
        "spot_iv_coupling": {
            "weight": 0.20,
            "metrics": {
                "corr_spot_div_diff": {"tolerance": 0.30, "weight": 1.0},
            },
        },
        "strategy_pnl_parity": {
            "weight": 0.30,
            "metrics": {
                "return_quantile_diff": {"tolerance": 0.02, "weight": 0.5},
                "ks": {"tolerance": 0.20, "weight": 0.2},
                "es_1pct_diff": {"tolerance": 0.03, "weight": 0.2},
                "max_dd_diff": {"tolerance": 0.10, "weight": 0.1},
            },
        },
    }

    return {
        "version": "mvp-v1",
        "strategies": strategies,
        "components": components,
    }
