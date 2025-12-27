import pytest

from src.llm.router import LLMAuthError, LLMRetryableError, LLMRouter


class DummyProvider:
    def __init__(self, *, result=None, error=None) -> None:
        self.result = result
        self.error = error
        self.calls = 0

    def call(self, messages, *, model, json_schema=None, tool_schema=None) -> str:
        self.calls += 1
        if self.error:
            raise self.error
        return self.result


def test_router_fallback_on_retryable_error() -> None:
    providers = {
        "p1": DummyProvider(error=LLMRetryableError("timeout")),
        "p2": DummyProvider(result="ok"),
    }
    router = LLMRouter(
        model_chains={"trading_optimist": ["p1:model-a", "p2:model-b"]},
        providers=providers,
    )

    result = router.call("trading_optimist", messages=[{"role": "user", "content": "hi"}])
    assert result == "ok"
    assert providers["p1"].calls == 1
    assert providers["p2"].calls == 1


def test_router_auth_error_stops_fallback() -> None:
    providers = {
        "p1": DummyProvider(error=LLMAuthError("nope")),
        "p2": DummyProvider(result="ok"),
    }
    router = LLMRouter(
        model_chains={"trading_optimist": ["p1:model-a", "p2:model-b"]},
        providers=providers,
    )

    with pytest.raises(LLMAuthError):
        router.call("trading_optimist", messages=[{"role": "user", "content": "hi"}])

    assert providers["p1"].calls == 1
    assert providers["p2"].calls == 0
