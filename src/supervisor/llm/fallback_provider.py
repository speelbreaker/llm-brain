from __future__ import annotations

from typing import Iterable

from .base import LLMProvider


class FallbackLLMProvider(LLMProvider):
    """Try providers in order until one succeeds."""

    def __init__(self, providers: Iterable[LLMProvider], models: Iterable[str | None] | None = None):
        self.providers = list(providers)
        self.models = list(models) if models is not None else [None] * len(self.providers)
        if len(self.models) < len(self.providers):
            # Pad overrides so lookups are safe even if fewer models are supplied.
            self.models.extend([None] * (len(self.providers) - len(self.models)))

    def _model_for_index(self, index: int, provided_model: str) -> str:
        override = self.models[index] if index < len(self.models) else None
        return override or provided_model

    async def generate(
        self,
        prompt: str,
        model: str,
        max_tokens: int = 500,
        temperature: float = 0.7,
    ) -> str:
        last_error: Exception | None = None
        for idx, provider in enumerate(self.providers):
            try:
                target_model = self._model_for_index(idx, model)
                return await provider.generate(
                    prompt,
                    target_model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            except Exception as exc:  # pragma: no cover - error path is validated via tests
                last_error = exc
        raise last_error or RuntimeError("No provider succeeded")

    async def generate_json(
        self,
        prompt: str,
        model: str,
        schema_hint: str | None,
        max_tokens: int = 500,
        temperature: float = 0.3,
    ) -> dict:
        last_error: Exception | None = None
        for idx, provider in enumerate(self.providers):
            try:
                target_model = self._model_for_index(idx, model)
                return await provider.generate_json(
                    prompt,
                    target_model,
                    schema_hint,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            except Exception as exc:  # pragma: no cover - error path is validated via tests
                last_error = exc
        raise last_error or RuntimeError("No provider succeeded")

    async def close(self) -> None:
        for provider in self.providers:
            try:
                await provider.close()
            except Exception:
                pass
