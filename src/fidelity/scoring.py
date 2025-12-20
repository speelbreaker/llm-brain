from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

from .metrics import exp_score


@dataclass(frozen=True)
class FidelityGates:
    trusted_overall: float = 80.0
    trusted_strategy_parity: float = 75.0
    trusted_tail_parity: float = 70.0

    warning_floor: float = 65.0


def gate_label(
    *,
    overall_score: float,
    strategy_parity_score: float,
    tail_parity_score: float,
    gates: Optional[FidelityGates] = None,
) -> str:
    g = gates or FidelityGates()

    if (
        overall_score >= g.trusted_overall
        and strategy_parity_score >= g.trusted_strategy_parity
        and tail_parity_score >= g.trusted_tail_parity
    ):
        return "TRUSTED"

    if overall_score >= g.warning_floor:
        return "WARNING"

    return "UNTRUSTED"


def score_components(components: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Compute component-level and overall scores.

    Input format (P0):
      components[name] = {
        "metrics": {metric_name: {"error": x, "tolerance": y, "k": 1.0, "weight": 1.0}},
        "weight": 0.2
      }

    Returns {"component_scores": {...}, "overall_score": ...}
    """
    component_scores: Dict[str, float] = {}

    overall_num = 0.0
    overall_den = 0.0

    for name, comp in components.items():
        metrics = comp.get("metrics") or {}
        comp_weight = float(comp.get("weight") or 0.0)

        num = 0.0
        den = 0.0
        for mname, m in metrics.items():
            w = float(m.get("weight") or 1.0)
            err = float(m.get("error") or 0.0)
            tol = float(m.get("tolerance") or 1.0)
            k = float(m.get("k") or 1.0)
            s = exp_score(err, tol, k=k)
            num += w * s
            den += w

        comp_score = (num / den) if den > 0 else 0.0
        component_scores[name] = float(comp_score)

        overall_num += comp_weight * comp_score
        overall_den += comp_weight

    overall_score = (overall_num / overall_den) if overall_den > 0 else 0.0

    return {
        "component_scores": component_scores,
        "overall_score": float(overall_score),
    }


def score_fidelity_components(
    *,
    components: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Score components with partial availability.

    Expected per component:
      {
        "weight": 0.2,
        "status": "ok"|"not_available",
        "metrics": {metric_name: {"error": x, "tolerance": y, "k": 1.0, "weight": 1.0}},
        "meta": {...}
      }

    If status != ok, weight is redistributed proportionally across ok components.
    """
    ok_items: list[Tuple[str, Dict[str, Any]]] = []
    na_items: list[str] = []
    for name, comp in components.items():
        status = (comp.get("status") or "ok").lower()
        if status == "ok":
            ok_items.append((name, comp))
        else:
            na_items.append(name)

    # Compute raw component scores only for ok components.
    raw_ok = score_components({k: v for k, v in ok_items}) if ok_items else {"component_scores": {}, "overall_score": 0.0}
    comp_scores: Dict[str, float] = {k: 0.0 for k in components}
    comp_scores.update({k: float(v) for k, v in (raw_ok.get("component_scores") or {}).items()})

    # Redistribute weights.
    total_weight_ok = sum(float(comp.get("weight") or 0.0) for _, comp in ok_items)
    if total_weight_ok <= 0:
        return {
            "overall_score": 0.0,
            "component_scores": comp_scores,
            "component_status": {k: (components[k].get("status") or "ok") for k in components},
            "redistributed_weights": {},
        }

    redistributed: Dict[str, float] = {}
    for name, comp in components.items():
        w = float(comp.get("weight") or 0.0)
        if (comp.get("status") or "ok").lower() != "ok":
            redistributed[name] = 0.0
        else:
            redistributed[name] = float(w / total_weight_ok)

    overall = 0.0
    for name, comp in ok_items:
        overall += redistributed[name] * float(comp_scores.get(name, 0.0))
    overall = float(overall)

    return {
        "overall_score": overall,
        "component_scores": comp_scores,
        "component_status": {k: (components[k].get("status") or "ok") for k in components},
        "redistributed_weights": redistributed,
        "not_available": na_items,
    }

