from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, List, Optional

from .market_replay import MarketReplay
from .canonical_strategies import StrategySpec, run_strategy


@dataclass(frozen=True)
class StrategyParityResult:
    strategy_name: str
    live: Dict[str, Any]
    synthetic: Dict[str, Any]


def run_parity_suite(
    *,
    decision_times: List[datetime],
    strategies: List[StrategySpec],
    live_market: MarketReplay,
    synthetic_market: MarketReplay,

) -> Dict[str, Any]:
    """Run the strategy parity suite.

    Executes the same canonical strategies on live vs synthetic market replays and
    returns a structured payload for reporting.
    """

    results: Dict[str, Any] = {
        "decision_times": [t.isoformat() for t in decision_times],
        "strategies": [],
    }

    for spec in strategies:
        live_snaps = [live_market.snapshot(t) for t in decision_times]
        synth_snaps = [synthetic_market.snapshot(t) for t in decision_times]

        live_trades = run_strategy(spec=spec, snapshots=live_snaps)
        synth_trades = run_strategy(spec=spec, snapshots=synth_snaps)

        live_valid = [t for t in live_trades if getattr(t, "is_valid", True)]
        synth_valid = [t for t in synth_trades if getattr(t, "is_valid", True)]

        results["strategies"].append(
            {
                "name": spec.name,
                "live": {
                    "num_trades": len(live_trades),
                    "valid_trades": len(live_valid),
                    "coverage_ratio": (len(live_valid) / max(len(live_trades), 1)),
                },
                "synthetic": {
                    "num_trades": len(synth_trades),
                    "valid_trades": len(synth_valid),
                    "coverage_ratio": (len(synth_valid) / max(len(synth_trades), 1)),
                },
            }
        )

    return results
