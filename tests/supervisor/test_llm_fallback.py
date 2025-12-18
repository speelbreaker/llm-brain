from types import SimpleNamespace

import pytest

from src.supervisor.config import SupervisorSettings
from src.supervisor.llm.fallback_provider import FallbackLLMProvider
from src.supervisor.llm.openai_provider import OpenAIProvider
from src.supervisor.llm.router import cleanup_providers, get_provider_for_role


class StubChatCompletions:
    def __init__(self):
        self.calls: list[dict] = []

    async def create(self, model, messages, response_format=None, max_tokens=None, temperature=None):
        self.calls.append(
            {
                "model": model,
                "response_format": response_format,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        )
        if response_format is not None:
            raise Exception("response_format json_object not supported by this model")

        return SimpleNamespace(
            choices=[
                SimpleNamespace(message=SimpleNamespace(content='{"ok": true}')),
            ]
        )


class StubClient:
    def __init__(self):
        self.completions = StubChatCompletions()
        self.chat = SimpleNamespace(completions=self.completions)
        self.closed = False

    async def close(self):
        self.closed = True


@pytest.mark.asyncio
async def test_openai_json_fallback_retries_without_response_format():
    provider = OpenAIProvider(api_key="test")
    stub = StubClient()
    provider.client = stub

    result = await provider.generate_json("hi", model="gpt-fake", max_tokens=10, temperature=0)

    assert result == {"ok": True}
    assert len(stub.completions.calls) == 2
    assert stub.completions.calls[0]["response_format"] == {"type": "json_object"}
    assert stub.completions.calls[1]["response_format"] is None

    await provider.close()
    assert stub.closed


@pytest.mark.asyncio
async def test_router_uses_fallback_only_when_gemini_key_set():
    await cleanup_providers()
    settings = SupervisorSettings(
        openai_api_key="openai-key",
        gemini_api_key="gem-key",
        gemini_model="gemini-1.5-pro",
        model_optimist="gpt-4o-mini",
    )

    provider, model = get_provider_for_role("optimist", settings)

    assert isinstance(provider, FallbackLLMProvider)
    assert model == "gpt-4o-mini"
    assert provider.models[1] == settings.gemini_model
    assert any(p.__class__.__name__ == "GeminiProvider" for p in provider.providers)

    await cleanup_providers()


@pytest.mark.asyncio
async def test_router_defaults_to_openai_without_gemini_key():
    await cleanup_providers()
    settings = SupervisorSettings(
        openai_api_key="openai-key",
        model_optimist="gpt-4o-mini",
    )

    provider, model = get_provider_for_role("optimist", settings)

    assert isinstance(provider, OpenAIProvider)
    assert model == "gpt-4o-mini"

    await cleanup_providers()
