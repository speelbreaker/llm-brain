"""Telegram webhook endpoint for chat continuity via OpenAI Conversations."""
from __future__ import annotations

import json
import logging
from typing import Any, Iterable

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from src.config import settings
from src.telegram.store import TelegramConversationStore


router = APIRouter()
logger = logging.getLogger("telegram_webhook")


def _log_event(event: str, **fields: Any) -> None:
    payload = {"event": event, **fields}
    logger.info(json.dumps(payload))


def _parse_allowed_chat_ids(raw: str) -> set[int]:
    allowed: set[int] = set()
    if not raw:
        return allowed
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            allowed.add(int(part))
        except ValueError:
            continue
    return allowed


def _extract_message(update: dict[str, Any]) -> dict[str, Any] | None:
    return update.get("message") or update.get("edited_message") or update.get("channel_post")


def _extract_text(message: dict[str, Any]) -> str:
    return str(message.get("text") or message.get("caption") or "").strip()


def _extract_response_text(payload: dict[str, Any]) -> str:
    output_text = payload.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()
    for item in payload.get("output", []) or []:
        if item.get("type") != "message":
            continue
        for part in item.get("content", []) or []:
            if part.get("type") in ("output_text", "text"):
                text = part.get("text") or part.get("value")
                if isinstance(text, str) and text.strip():
                    return text.strip()
    choices = payload.get("choices")
    if isinstance(choices, Iterable):
        for choice in choices:
            message = choice.get("message") if isinstance(choice, dict) else None
            if message and isinstance(message.get("content"), str):
                content = message["content"].strip()
                if content:
                    return content
    return ""


async def _openai_create_conversation(http_client: httpx.AsyncClient) -> str:
    response = await http_client.post(
        "https://api.openai.com/v1/conversations",
        headers={"Authorization": f"Bearer {settings.openai_api_key}"},
        json={"metadata": {"source": "telegram"}},
    )
    response.raise_for_status()
    payload = response.json()
    conversation_id = payload.get("id")
    if not conversation_id:
        raise RuntimeError("OpenAI conversation creation did not return an id")
    return conversation_id


async def _openai_add_message(
    http_client: httpx.AsyncClient,
    conversation_id: str,
    role: str,
    text: str,
) -> None:
    response = await http_client.post(
        f"https://api.openai.com/v1/conversations/{conversation_id}/messages",
        headers={"Authorization": f"Bearer {settings.openai_api_key}"},
        json={
            "role": role,
            "content": [{"type": "text", "text": text}],
        },
    )
    response.raise_for_status()


async def _openai_create_response(
    http_client: httpx.AsyncClient,
    conversation_id: str,
    user_text: str,
) -> str:
    response = await http_client.post(
        "https://api.openai.com/v1/responses",
        headers={"Authorization": f"Bearer {settings.openai_api_key}"},
        json={
            "model": settings.openai_model,
            "conversation_id": conversation_id,
            "input": user_text,
        },
    )
    response.raise_for_status()
    return _extract_response_text(response.json())


async def _send_telegram_message(
    http_client: httpx.AsyncClient,
    chat_id: int,
    text: str,
) -> None:
    response = await http_client.post(
        f"https://api.telegram.org/bot{settings.telegram_bot_token}/sendMessage",
        json={"chat_id": chat_id, "text": text},
    )
    response.raise_for_status()


@router.post("/telegram/webhook/{path_secret}")
async def telegram_webhook(path_secret: str, request: Request) -> JSONResponse:
    """Handle Telegram updates. Returns 403 for disallowed chat IDs."""
    if not settings.telegram_webhook_path_secret or path_secret != settings.telegram_webhook_path_secret:
        return JSONResponse(status_code=404, content={"ok": False, "error": "not_found"})

    try:
        update = await request.json()
    except Exception:
        _log_event("telegram_webhook_invalid_json")
        return JSONResponse(status_code=200, content={"ok": False, "error": "invalid_json"})

    message = _extract_message(update)
    if not message:
        _log_event("telegram_webhook_no_message", update_id=update.get("update_id"))
        return JSONResponse(status_code=200, content={"ok": True, "status": "ignored"})

    chat = message.get("chat") or {}
    chat_id = chat.get("id")
    text = _extract_text(message)
    if not chat_id or not text:
        _log_event(
            "telegram_webhook_missing_fields",
            update_id=update.get("update_id"),
            has_chat_id=bool(chat_id),
            has_text=bool(text),
        )
        return JSONResponse(status_code=200, content={"ok": True, "status": "ignored"})

    allowed_chat_ids = _parse_allowed_chat_ids(settings.telegram_allowed_chat_ids)
    if allowed_chat_ids and int(chat_id) not in allowed_chat_ids:
        _log_event("telegram_webhook_forbidden", chat_id=chat_id)
        return JSONResponse(status_code=403, content={"ok": False, "error": "forbidden"})

    if not settings.openai_api_key:
        _log_event("telegram_webhook_missing_openai_key", chat_id=chat_id)
        return JSONResponse(status_code=200, content={"ok": False, "error": "missing_openai_key"})

    if not settings.telegram_bot_token:
        _log_event("telegram_webhook_missing_bot_token", chat_id=chat_id)
        return JSONResponse(status_code=200, content={"ok": False, "error": "missing_bot_token"})

    store = TelegramConversationStore()
    conversation_state = store.get(int(chat_id))
    bootstrap_needed = False

    async with httpx.AsyncClient(timeout=httpx.Timeout(20.0)) as http_client:
        try:
            if conversation_state is None:
                conversation_id = await _openai_create_conversation(http_client)
                store.create(int(chat_id), conversation_id)
                bootstrap_needed = True
            else:
                conversation_id = conversation_state.conversation_id
                bootstrap_needed = not conversation_state.bootstrap_applied

            if bootstrap_needed and settings.telegram_bootstrap_context:
                await _openai_add_message(
                    http_client,
                    conversation_id,
                    role="system",
                    text=settings.telegram_bootstrap_context,
                )
                store.mark_bootstrap_applied(int(chat_id))

            response_text = await _openai_create_response(http_client, conversation_id, text)
            if not response_text:
                response_text = "Sorry, I could not generate a response."

            await _send_telegram_message(http_client, int(chat_id), response_text)
        except Exception as exc:
            error_payload = {"error_type": type(exc).__name__}
            if isinstance(exc, httpx.HTTPStatusError):
                error_payload["status_code"] = exc.response.status_code
            _log_event(
                "telegram_webhook_error",
                chat_id=chat_id,
                **error_payload,
            )
            return JSONResponse(status_code=200, content={"ok": False, "error": "processing_error"})

    _log_event("telegram_webhook_ok", chat_id=chat_id)
    return JSONResponse(status_code=200, content={"ok": True})
