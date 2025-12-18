"""LLM provider router for multi-provider support."""
from typing import Literal

from ..config import SupervisorSettings
from .base import LLMProvider
from .fallback_provider import FallbackLLMProvider
from .gemini_provider import GeminiProvider
from .openai_provider import OpenAIProvider

_provider_cache: dict[str, LLMProvider] = {}


def get_provider_for_role(
    role: Literal["optimist", "skeptic", "arbiter"],
    settings: SupervisorSettings,
) -> tuple[LLMProvider, str]:
    """Return provider + model for the requested debate role."""
    if role == "optimist":
        model = settings.model_optimist
    elif role == "skeptic":
        model = settings.model_skeptic
    else:
        model = settings.model_arbiter

    provider = _get_or_create_provider(settings)
    return provider, model


def _get_or_create_provider(settings: SupervisorSettings) -> LLMProvider:
    """Get or create a provider instance (cached)."""
    cache_key = "fallback" if settings.gemini_api_key else "openai"

    if cache_key in _provider_cache:
        return _provider_cache[cache_key]

    openai = OpenAIProvider(api_key=settings.openai_api_key)
    if settings.gemini_api_key:
        gemini = GeminiProvider(
            api_key=settings.gemini_api_key,
            model=settings.gemini_model,
        )
        provider = FallbackLLMProvider(
            [openai, gemini],
            models=[None, settings.gemini_model],
        )
    else:
        provider = openai

    _provider_cache[cache_key] = provider
    return provider


async def cleanup_providers() -> None:
    """Cleanup all cached providers."""
    for provider in _provider_cache.values():
        await provider.close()
    _provider_cache.clear()
