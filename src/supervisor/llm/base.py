"""Base interface for LLM providers."""

from abc import ABC, abstractmethod
from typing import Literal, Optional

from pydantic import BaseModel, Field


class DebateResponse(BaseModel):
    """Strict JSON schema for debate agent responses."""
    role: Literal["optimist", "skeptic", "arbiter"]
    summary: str
    bullets: list[str] = Field(default_factory=list, max_length=3)
    auto_fix_allowed: Optional[bool] = None
    objectives: Optional[list[str]] = None
    risk_level: Optional[Literal["low", "med", "high"]] = None
    stop_reason: Optional[str] = None


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        model: str,
        max_tokens: int = 500,
        temperature: float = 0.7,
    ) -> str:
        """Generate a completion from the LLM."""
        pass
    
    @abstractmethod
    async def generate_json(
        self,
        prompt: str,
        model: str,
        schema_hint: str,
        max_tokens: int = 500,
        temperature: float = 0.3,
    ) -> dict:
        """Generate a JSON response from the LLM with schema enforcement."""
        pass
    
    async def close(self) -> None:
        """Cleanup resources (optional override)."""
        pass
