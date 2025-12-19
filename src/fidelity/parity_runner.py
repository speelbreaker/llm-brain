from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, List, Optional

from .market_replay import MarketReplay
from .strategies_canonical import StrategySpec


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

    P0: skeleton that proves wiring (same timestamps, two markets, report shape).
    Next steps will implement strategy execution and real metrics.
    """

    results: Dict[str, Any] = {
        "decision_times": [t.isoformat() for t in decision_times],
        "strategies": [],
    }

    # Minimal smoke check: confirm both markets can snapshot all decision times.
    live_spots = []
    synth_spots = []
    for t in decision_times:
        live_snap = live_market.snapshot(t)
        synth_snap = synthetic_market.snapshot(t)
        live_spots.append(live_snap.spot)
        synth_spots.append(synth_snap.spot)

    for spec in strategies:
        results["strategies"].append(
            {
                "name": spec.name,
                "live": {
                    "num_trades": 0,
                    "notes": "P0 placeholder (execution not implemented)",
                    "spot_first": live_spots[0] if live_spots else None,
                    "spot_last": live_spots[-1] if live_spots else None,
                },
                "synthetic": {
                    "num_trades": 0,
                    "notes": "P0 placeholder (execution not implemented)",
                    "spot_first": synth_spots[0] if synth_spots else None,
                    "spot_last": synth_spots[-1] if synth_spots else None,
                },
            }
        )

    return results
