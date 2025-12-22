"""Database models for Telegram conversation mapping."""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import Column, DateTime, Integer, String, UniqueConstraint

from src.db import Base


class TelegramConversation(Base):
    """Maps Telegram chat IDs to OpenAI conversation IDs."""

    __tablename__ = "telegram_conversations"
    __table_args__ = (UniqueConstraint("telegram_chat_id", name="uq_telegram_chat_id"),)

    id = Column(Integer, primary_key=True, index=True)
    telegram_chat_id = Column(Integer, nullable=False)
    openai_conversation_id = Column(String, nullable=False)
    bootstrap_applied = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
