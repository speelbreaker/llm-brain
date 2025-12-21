"""Fidelity API routes.

These endpoints are intentionally file-based so they work without requiring
database access.

Note: This router exposes exactly two endpoints:
- GET /api/fidelity/latest
- GET /api/fidelity/history?limit=30
"""

from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

router = APIRouter()


@router.get("/api/fidelity/latest")
def get_fidelity_latest() -> JSONResponse:
    """Return the latest Synthetic Fidelity summary.

    404 if no runs exist.
    """
    from src.backtest import fidelity_store

    latest = fidelity_store.load_latest()
    if not latest:
        return JSONResponse(status_code=404, content={"error": "no_fidelity_runs"})

    # latest.json is already a summary payload.
    summary: Dict[str, Any] = dict(latest)
    return JSONResponse(content=summary)


@router.get("/api/fidelity/history")
def get_fidelity_history(
    limit: int = Query(30, ge=1, le=200),
) -> JSONResponse:
    """Return recent Synthetic Fidelity summaries newest->oldest."""
    from src.backtest import fidelity_store

    runs: List[Dict[str, Any]] = fidelity_store.load_history(limit=int(limit))
    return JSONResponse(content=runs)
