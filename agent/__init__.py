"""
Telegram Code Review Agent

A code review bot that monitors repository changes and provides
structured reviews via Telegram commands.
"""
from agent.config import settings
from agent.storage import init_db

__all__ = ["settings", "init_db"]
