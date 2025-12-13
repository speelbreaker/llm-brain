"""
Configuration module for the Telegram Code Review Agent.

Reads settings from environment variables (Replit Secrets).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Set


def _parse_allowed_ids(value: Optional[str]) -> Set[int]:
    """Parse comma-separated list of Telegram user IDs."""
    if not value:
        return set()
    ids = set()
    for part in value.split(","):
        part = part.strip()
        if part.isdigit():
            ids.add(int(part))
    return ids


@dataclass
class Settings:
    """Agent configuration settings."""
    
    telegram_bot_token: str
    allowed_user_ids: Set[int] = field(default_factory=set)
    openai_api_key: Optional[str] = None
    
    openai_model_review: str = "gpt-5.2-pro"
    openai_model_fast: str = "gpt-5.2"
    openai_reasoning_effort: str = "high"
    
    db_path: Path = field(default_factory=lambda: Path("data/agent_data.db"))
    log_paths: list[str] = field(default_factory=lambda: ["logs/app.log", "logs/tests.log"])
    config_watch_files: list[str] = field(default_factory=lambda: [
        ".env.example",
        "src/config.py",
        "pyproject.toml",
        "replit.nix",
        "requirements.txt",
    ])
    
    max_diff_lines: int = 500
    max_diff_chars: int = 50000
    
    @classmethod
    def from_env(cls) -> "Settings":
        """Load settings from environment variables."""
        token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        if not token:
            raise ValueError("TELEGRAM_BOT_TOKEN environment variable is required")
        
        allowed_ids = _parse_allowed_ids(os.environ.get("TELEGRAM_ALLOWED_USER_IDS"))
        openai_key = os.environ.get("AI_INTEGRATIONS_OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
        
        model_review = os.environ.get("OPENAI_MODEL_REVIEW", "gpt-5.2-pro")
        model_fast = os.environ.get("OPENAI_MODEL_FAST", "gpt-5.2")
        reasoning_effort = os.environ.get("OPENAI_REASONING_EFFORT", "high")
        
        return cls(
            telegram_bot_token=token,
            allowed_user_ids=allowed_ids,
            openai_api_key=openai_key,
            openai_model_review=model_review,
            openai_model_fast=model_fast,
            openai_reasoning_effort=reasoning_effort,
        )
    
    def is_user_allowed(self, user_id: int) -> bool:
        """Check if a Telegram user is authorized."""
        if not self.allowed_user_ids:
            return False
        return user_id in self.allowed_user_ids


def _load_settings() -> Optional[Settings]:
    """Attempt to load settings, return None if not configured."""
    try:
        return Settings.from_env()
    except ValueError:
        return None


settings: Optional[Settings] = _load_settings()
