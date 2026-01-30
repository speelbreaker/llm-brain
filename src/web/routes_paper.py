"""Paper portfolio API endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from src.config import settings
from src.paper_portfolios import get_tracker

router = APIRouter(tags=["paper"])




@router.get("/api/paper/positions/open")
def paper_open_positions(request: Request, lane: str = "rule") -> JSONResponse:
    if not settings.paper_compare_enabled:
        return JSONResponse(status_code=404, content={"ok": False, "error": "not_found"})
    if not settings.paper_compare_enabled:
        return JSONResponse(status_code=404, content={"ok": False, "error": "not_found"})
    lane = (lane or "rule").strip().lower()
    if lane not in ("rule", "llm", "debate"):
        return JSONResponse(status_code=400, content={"ok": False, "error": "lane must be rule|llm|debate"})
    payload = get_tracker(lane).get_open_positions_payload(include_sandbox=True)
    return JSONResponse(content={"ok": True, "lane": lane, **payload})


@router.get("/api/paper/positions/closed")
def paper_closed_positions(request: Request, lane: str = "rule") -> JSONResponse:
    lane = (lane or "rule").strip().lower()
    if lane not in ("rule", "llm", "debate"):
        return JSONResponse(status_code=400, content={"ok": False, "error": "lane must be rule|llm|debate"})
    payload = get_tracker(lane).get_closed_positions_payload(include_sandbox=True)
    return JSONResponse(content={"ok": True, "lane": lane, **payload})
