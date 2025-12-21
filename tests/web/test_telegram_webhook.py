"""Endpoint tests for Telegram webhook integration."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.config import settings
from src.telegram.store import TelegramConversationStore
from src.web_app import app
import src.web.routes_telegram as routes_telegram


@pytest.fixture
def client():
    return TestClient(app)


def _update_payload(chat_id: int, text: str, update_id: int = 1) -> dict:
    return {
        "update_id": update_id,
        "message": {
            "message_id": 10,
            "chat": {"id": chat_id, "type": "private"},
            "text": text,
        },
    }


def _configure_settings(tmp_path, allowed_ids: str) -> None:
    settings.telegram_webhook_path_secret = "secret"
    settings.telegram_allowed_chat_ids = allowed_ids
    settings.openai_api_key = "test-openai-key"
    settings.openai_model = "gpt-4.1"
    settings.telegram_bot_token = "test-telegram-token"
    settings.telegram_bootstrap_context = "bootstrap"
    settings.telegram_store_path = str(tmp_path / "telegram_store.json")


def test_webhook_allowed_chat_creates_mapping(client, monkeypatch, tmp_path):
    _configure_settings(tmp_path, allowed_ids="123")

    calls = {"create_conversation": 0, "add_message": 0, "create_response": 0, "send_message": 0}

    async def fake_create_conversation(http_client):
        calls["create_conversation"] += 1
        return "conv_1"

    async def fake_add_message(http_client, conversation_id, role, text):
        calls["add_message"] += 1

    async def fake_create_response(http_client, conversation_id, user_text):
        calls["create_response"] += 1
        return "hello"

    async def fake_send_message(http_client, chat_id, text):
        calls["send_message"] += 1

    monkeypatch.setattr(routes_telegram, "_openai_create_conversation", fake_create_conversation)
    monkeypatch.setattr(routes_telegram, "_openai_add_message", fake_add_message)
    monkeypatch.setattr(routes_telegram, "_openai_create_response", fake_create_response)
    monkeypatch.setattr(routes_telegram, "_send_telegram_message", fake_send_message)

    response = client.post("/telegram/webhook/secret", json=_update_payload(123, "hi"))
    assert response.status_code == 200

    store = TelegramConversationStore()
    state = store.get(123)
    assert state is not None
    assert state.conversation_id == "conv_1"
    assert state.bootstrap_applied is True

    assert calls["create_conversation"] == 1
    assert calls["add_message"] == 1
    assert calls["create_response"] == 1
    assert calls["send_message"] == 1


def test_webhook_disallowed_chat_returns_403(client, monkeypatch, tmp_path):
    _configure_settings(tmp_path, allowed_ids="999")

    async def should_not_call(*args, **kwargs):
        raise AssertionError("OpenAI/Telegram should not be called for disallowed chats")

    monkeypatch.setattr(routes_telegram, "_openai_create_conversation", should_not_call)
    monkeypatch.setattr(routes_telegram, "_openai_add_message", should_not_call)
    monkeypatch.setattr(routes_telegram, "_openai_create_response", should_not_call)
    monkeypatch.setattr(routes_telegram, "_send_telegram_message", should_not_call)

    response = client.post("/telegram/webhook/secret", json=_update_payload(123, "blocked"))
    assert response.status_code == 403


def test_bootstrap_only_once(client, monkeypatch, tmp_path):
    _configure_settings(tmp_path, allowed_ids="123")

    calls = {"create_conversation": 0, "add_message": 0, "create_response": 0, "send_message": 0}

    async def fake_create_conversation(http_client):
        calls["create_conversation"] += 1
        return "conv_2"

    async def fake_add_message(http_client, conversation_id, role, text):
        calls["add_message"] += 1

    async def fake_create_response(http_client, conversation_id, user_text):
        calls["create_response"] += 1
        return "ok"

    async def fake_send_message(http_client, chat_id, text):
        calls["send_message"] += 1

    monkeypatch.setattr(routes_telegram, "_openai_create_conversation", fake_create_conversation)
    monkeypatch.setattr(routes_telegram, "_openai_add_message", fake_add_message)
    monkeypatch.setattr(routes_telegram, "_openai_create_response", fake_create_response)
    monkeypatch.setattr(routes_telegram, "_send_telegram_message", fake_send_message)

    response = client.post("/telegram/webhook/secret", json=_update_payload(123, "first"))
    assert response.status_code == 200

    response = client.post("/telegram/webhook/secret", json=_update_payload(123, "second", update_id=2))
    assert response.status_code == 200

    assert calls["create_conversation"] == 1
    assert calls["add_message"] == 1
    assert calls["create_response"] == 2
    assert calls["send_message"] == 2
