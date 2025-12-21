"""Storage for Telegram chat to OpenAI conversation mappings."""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy import select

from src.config import settings
from src.db import SessionLocal
from src.db.models_telegram import TelegramConversation


@dataclass
class TelegramConversationState:
    conversation_id: str
    bootstrap_applied: bool


class TelegramConversationStore:
    """Persist Telegram chat mappings in SQLAlchemy or a JSON store."""

    def __init__(self, json_path: str | None = None) -> None:
        self._json_path = Path(json_path or settings.telegram_store_path)
        self._lock = threading.Lock()

    @property
    def _use_db(self) -> bool:
        return SessionLocal is not None

    def get(self, chat_id: int) -> TelegramConversationState | None:
        if self._use_db:
            with SessionLocal() as session:
                row = session.execute(
                    select(TelegramConversation).where(TelegramConversation.telegram_chat_id == chat_id)
                ).scalar_one_or_none()
                if not row:
                    return None
                return TelegramConversationState(
                    conversation_id=row.openai_conversation_id,
                    bootstrap_applied=bool(row.bootstrap_applied),
                )
        data = self._load_json()
        record = data.get(str(chat_id))
        if not record:
            return None
        return TelegramConversationState(
            conversation_id=str(record.get("conversation_id", "")),
            bootstrap_applied=bool(record.get("bootstrap_applied", False)),
        )

    def create(self, chat_id: int, conversation_id: str) -> None:
        if self._use_db:
            with SessionLocal() as session:
                row = TelegramConversation(
                    telegram_chat_id=chat_id,
                    openai_conversation_id=conversation_id,
                    bootstrap_applied=0,
                )
                session.add(row)
                session.commit()
            return
        with self._lock:
            data = self._load_json()
            data[str(chat_id)] = {
                "conversation_id": conversation_id,
                "bootstrap_applied": False,
            }
            self._save_json(data)

    def mark_bootstrap_applied(self, chat_id: int) -> None:
        if self._use_db:
            with SessionLocal() as session:
                row = session.execute(
                    select(TelegramConversation).where(TelegramConversation.telegram_chat_id == chat_id)
                ).scalar_one_or_none()
                if not row:
                    return
                row.bootstrap_applied = 1
                session.commit()
            return
        with self._lock:
            data = self._load_json()
            record = data.get(str(chat_id))
            if not record:
                return
            record["bootstrap_applied"] = True
            data[str(chat_id)] = record
            self._save_json(data)

    def _load_json(self) -> dict[str, Any]:
        if not self._json_path.exists():
            return {}
        try:
            with self._json_path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except (json.JSONDecodeError, OSError):
            return {}

    def _save_json(self, data: dict[str, Any]) -> None:
        self._json_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self._json_path.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle)
        tmp_path.replace(self._json_path)
