"""LLM provider router for multi-provider support."""

from typing import Literal

from ..config import SupervisorSettings
from .base import LLMProvider
from .openai_provider import OpenAIProvider
from .gemini_provider import GeminiProvider


_provider_cache: dict[str, LLMProvider] = {}


def get_provider_for_role(
    role: Literal["optimist", "skeptic", "arbiter"],
    settings: SupervisorSettings,
) -> tuple[LLMProvider, str]:
    """Get the appropriate LLM provider and model for a debate role.
    
    Returns:
        Tuple of (provider instance, model name)
    """
    if role == "optimist":
        provider_name = settings.optimist_provider.lower()
        model = settings.model_optimist
    elif role == "skeptic":
        provider_name = settings.skeptic_provider.lower()
        model = settings.model_skeptic
    else:
        provider_name = settings.arbiter_provider.lower()
        model = settings.model_arbiter
    
    provider = _get_or_create_provider(provider_name, settings)
    return provider, model


def _get_or_create_provider(
    provider_name: str,
    settings: SupervisorSettings,
) -> LLMProvider:
    """Get or create a provider instance (cached)."""
    cache_key = provider_name
    
    if cache_key in _provider_cache:
        return _provider_cache[cache_key]
    
    if provider_name == "gemini":
        if not settings.gemini_api_key:
            raise ValueError("GEMINI_API_KEY required when using Gemini provider")
        provider = GeminiProvider(
            api_key=settings.gemini_api_key,
            base_url=settings.gemini_base_url,
        )
    else:
        provider = OpenAIProvider(api_key=settings.openai_api_key)
    
    _provider_cache[cache_key] = provider
    return provider


def get_provider(provider_name: str, settings: SupervisorSettings) -> LLMProvider:
    """Get a provider instance by name."""
    return _get_or_create_provider(provider_name.lower(), settings)


async def cleanup_providers() -> None:
    """Cleanup all cached providers."""
    for provider in _provider_cache.values():
        await provider.close()
    _provider_cache.clear()
