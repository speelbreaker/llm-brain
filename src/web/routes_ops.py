"""Ops endpoints for fidelity and gates visibility."""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/ops", tags=["ops"])


@router.get("/health")
def get_ops_health():
    """Return current ops health artifact."""
    path = Path("docs/OPS_HEALTH_latest.json")
    if not path.exists():
        return JSONResponse(
            status_code=503,
            content={"error": "OPS_HEALTH_latest.json not found"},
        )
    try:
        return JSONResponse(content=json.loads(path.read_text()))
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to parse: {e}"},
        )


@router.get("/fidelity/{underlying}")
def get_ops_fidelity(underlying: str):
    """Return fidelity status for underlying."""
    u = underlying.upper()
    path = Path(f"docs/FIDELITY_{u}_latest.json")
    if not path.exists():
        return JSONResponse(
            status_code=404,
            content={"error": f"FIDELITY_{u}_latest.json not found"},
        )
    try:
        return JSONResponse(content=json.loads(path.read_text()))
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to parse: {e}"},
        )


@router.get("/gates")
def get_ops_gates():
    """Return combined can_trade decision with Truth/Trust/Trade breakdown."""
    from src.ops.trade_permission import get_current_trade_permission
    from src.config import settings

    perm = get_current_trade_permission(settings)

    fidelity_status = {}
    for u in getattr(settings, "underlyings", ["BTC", "ETH"]):
        path = Path(f"docs/FIDELITY_{u}_latest.json")
        if path.exists():
            try:
                data = json.loads(path.read_text())
                fidelity_status[u] = {
                    "can_trade": data.get("can_trade"),
                    "overall_status": data.get("overall_status"),
                }
            except Exception:
                fidelity_status[u] = {"error": "parse_failed"}

    return JSONResponse(
        content={
            "can_trade": perm.can_trade,
            "effective_trade_mode": perm.effective_trade_mode.value,
            "permission_code": perm.code.value,
            "reason": perm.reason,
            "breakdown": {
                "truth": {
                    "description": "Data availability checks",
                    "status": "check_fidelity_preflight",
                },
                "trust": {
                    "description": "Calibration and fidelity checks",
                    "fidelity": fidelity_status,
                },
                "trade": {
                    "description": "Trading mode and safety gates",
                    "kill_switch": settings.kill_switch_enabled,
                    "trade_mode": getattr(settings.trade_mode, "value", str(settings.trade_mode)),
                },
            },
        }
    )
