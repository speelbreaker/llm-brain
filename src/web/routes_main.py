"""Main routes for Options Trading Agent - status, chat, training, and strategy endpoints."""
from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Body
from fastapi.responses import JSONResponse

from src.status_store import status_store
from src.decisions_store import decisions_store
from src.chat_with_agent import chat_with_agent_full, get_chat_messages, clear_chat_history
from src.config import settings
from src.strategy_status import build_strategy_status, StrategyStatus
from src.rules_summary import build_rules_summary_from_settings

router = APIRouter()


@router.get("/status")
def get_status() -> JSONResponse:
    """Return the latest agent status snapshot."""
    data = status_store.get()
    return JSONResponse(content=data)


@router.get("/health")
def health_check() -> JSONResponse:
    """Health check endpoint for deployment."""
    return JSONResponse(content={"status": "healthy", "service": "options-trading-agent"})


@router.post("/chat")
def chat_endpoint(
    payload: Dict[str, Any] = Body(..., example={"question": "Why did you pick the 97k call?"}),
) -> JSONResponse:
    """Ask the agent a question about its recent behavior. Returns full conversation history."""
    question = payload.get("question", "").strip()
    if not question:
        return JSONResponse(
            status_code=400,
            content={"error": "Missing 'question' field in request body."},
        )

    try:
        result = chat_with_agent_full(question, log_limit=20)
        return JSONResponse(content={"question": question, "answer": result["answer"], "messages": result["messages"]})
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to generate answer: {str(e)}"},
        )


@router.get("/chat/messages")
def get_chat_history_endpoint() -> JSONResponse:
    """Get the full chat conversation history."""
    return JSONResponse(content={"messages": get_chat_messages()})


@router.post("/chat/clear")
def clear_chat_endpoint() -> JSONResponse:
    """Clear the chat conversation history."""
    clear_chat_history()
    return JSONResponse(content={"status": "cleared", "messages": []})


@router.get("/api/agent/decisions")
def get_agent_decisions() -> JSONResponse:
    """Return recent agent decisions for the dashboard."""
    decisions = decisions_store.get_all()
    last_update = decisions_store.get_last_update()
    
    return JSONResponse(content={
        "mode": "llm" if settings.llm_enabled else "rule_based",
        "llm_enabled": settings.llm_enabled,
        "dry_run": settings.dry_run,
        "training_mode": settings.is_training_enabled,
        "last_update": last_update.isoformat() if last_update else None,
        "decisions": decisions,
    })


@router.get("/api/training/status")
def get_training_status() -> JSONResponse:
    """Get current training mode status."""
    return JSONResponse(content={
        "enabled": settings.is_training_enabled,
        "training_mode": settings.training_mode,
        "strategies": settings.training_strategies,
        "is_research": settings.is_research,
        "dry_run": settings.dry_run,
    })


@router.post("/api/training/toggle")
def toggle_training_mode(payload: Dict[str, Any] = Body(...)) -> JSONResponse:
    """Toggle training mode on/off."""
    enable = payload.get("enable", False)
    
    if enable:
        if not settings.is_research:
            return JSONResponse(
                status_code=400,
                content={"error": "Training mode requires RESEARCH mode"},
            )
    
    settings.training_mode = enable
    
    return JSONResponse(content={
        "enabled": settings.is_training_enabled,
        "training_mode": settings.training_mode,
        "strategies": settings.training_strategies,
    })


@router.get("/api/strategy-status", response_model=StrategyStatus)
def get_strategy_status() -> JSONResponse:
    """
    Get current strategy and safeguards status for the UI.
    Shows mode, network, active rules, and safeguard states.
    """
    status = status_store.get() or {}
    config_snapshot = status.get("config_snapshot") or {}
    strategy_status = build_strategy_status(config_snapshot)
    return JSONResponse(content=strategy_status.model_dump())


@router.get("/api/rules-summary")
def get_rules_summary() -> JSONResponse:
    """Get the current rules summary for UI display."""
    summary = build_rules_summary_from_settings()
    return JSONResponse(content=summary)
