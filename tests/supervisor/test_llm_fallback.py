"""Tests for LLM fallback and circuit breaker behavior."""

import asyncio

import pytest

from src.supervisor.llm import fallback as llm_fallback


class DummyProvider:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    async def generate(self, prompt, model, max_tokens=500, temperature=0.7):
        self.calls += 1
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class DummySettings:
    model_optimist = "opt-model"
    model_skeptic = "skep-model"
    model_arbiter = "arb-model"
    openai_api_key = "dummy"
    gemini_api_key = "dummy"
    gemini_base_url = "https://example.test"


@pytest.mark.asyncio
async def test_fallback_to_gemini_on_retryable_openai_error(monkeypatch):
    openai_provider = DummyProvider([TimeoutError("timeout"), TimeoutError("timeout")])
    gemini_provider = DummyProvider(['{"role":"optimist","summary":"ok","bullets":["a"]}'])

    def fake_get_provider(name, settings):
        return openai_provider if name == "openai" else gemini_provider

    monkeypatch.setattr(llm_fallback, "get_provider", fake_get_provider)
    llm_fallback._breaker = llm_fallback.ProviderCircuitBreaker(failure_threshold=3, window_seconds=60, cooldown_seconds=60)

    def validator(payload):
        if payload.get("role") != "optimist":
            raise ValueError("invalid role")

    result, provider_used = await llm_fallback.generate_json_with_fallback(
        settings=DummySettings(),
        provider_chain=["openai", "gemini"],
        model_for_provider={"openai": "opt-model", "gemini": "opt-model"},
        prompt="test",
        schema_hint="{}",
        validator=validator,
        max_tokens=50,
        temperature=0.2,
        max_retries=2,
    )

    assert provider_used == "gemini"
    assert result["summary"] == "ok"
    assert openai_provider.calls == 2


@pytest.mark.asyncio
async def test_invalid_json_triggers_retry_then_fallback(monkeypatch):
    openai_provider = DummyProvider(["not-json", "still not json"])
    gemini_provider = DummyProvider(['{"role":"optimist","summary":"ok","bullets":["a"]}'])

    def fake_get_provider(name, settings):
        return openai_provider if name == "openai" else gemini_provider

    monkeypatch.setattr(llm_fallback, "get_provider", fake_get_provider)
    llm_fallback._breaker = llm_fallback.ProviderCircuitBreaker(failure_threshold=3, window_seconds=60, cooldown_seconds=60)

    def validator(payload):
        if payload.get("role") != "optimist":
            raise ValueError("invalid role")

    result, provider_used = await llm_fallback.generate_json_with_fallback(
        settings=DummySettings(),
        provider_chain=["openai", "gemini"],
        model_for_provider={"openai": "opt-model", "gemini": "opt-model"},
        prompt="test",
        schema_hint="{}",
        validator=validator,
        max_tokens=50,
        temperature=0.2,
        max_retries=2,
    )

    assert provider_used == "gemini"
    assert openai_provider.calls == 2


@pytest.mark.asyncio
async def test_circuit_breaker_opens(monkeypatch):
    openai_provider = DummyProvider([TimeoutError("timeout"), TimeoutError("timeout")])
    gemini_provider = DummyProvider(['{"role":"optimist","summary":"ok","bullets":["a"]}'])

    def fake_get_provider(name, settings):
        return openai_provider if name == "openai" else gemini_provider

    monkeypatch.setattr(llm_fallback, "get_provider", fake_get_provider)
    llm_fallback._breaker = llm_fallback.ProviderCircuitBreaker(
        failure_threshold=1, window_seconds=60, cooldown_seconds=60
    )

    def validator(payload):
        if payload.get("role") != "optimist":
            raise ValueError("invalid role")

    result, provider_used = await llm_fallback.generate_json_with_fallback(
        settings=DummySettings(),
        provider_chain=["openai", "gemini"],
        model_for_provider={"openai": "opt-model", "gemini": "opt-model"},
        prompt="test",
        schema_hint="{}",
        validator=validator,
        max_tokens=50,
        temperature=0.2,
        max_retries=1,
    )

    assert provider_used == "gemini"
    assert llm_fallback._breaker.is_open("openai")


@pytest.mark.asyncio
async def test_all_providers_unavailable(monkeypatch):
    openai_provider = DummyProvider([TimeoutError("timeout")])
    gemini_provider = DummyProvider([TimeoutError("timeout")])

    def fake_get_provider(name, settings):
        return openai_provider if name == "openai" else gemini_provider

    monkeypatch.setattr(llm_fallback, "get_provider", fake_get_provider)
    llm_fallback._breaker = llm_fallback.ProviderCircuitBreaker(
        failure_threshold=3, window_seconds=60, cooldown_seconds=60
    )

    def validator(payload):
        if payload.get("role") != "optimist":
            raise ValueError("invalid role")

    with pytest.raises(llm_fallback.LLMUnavailable):
        await llm_fallback.generate_json_with_fallback(
            settings=DummySettings(),
            provider_chain=["openai", "gemini"],
            model_for_provider={"openai": "opt-model", "gemini": "opt-model"},
            prompt="test",
            schema_hint="{}",
            validator=validator,
            max_tokens=50,
            temperature=0.2,
            max_retries=1,
        )
