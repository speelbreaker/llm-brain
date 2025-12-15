"""LLM provider abstraction layer."""

from .base import LLMProvider, DebateResponse
from .router import get_provider_for_role

__all__ = ["LLMProvider", "DebateResponse", "get_provider_for_role"]
