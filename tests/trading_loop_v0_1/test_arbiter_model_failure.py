from datetime import datetime, timezone

from src.llm.router import LLMRetryableError, LLMRouter
from src.trading_loop_v0_1.llm_agents import arbiter_decide
from src.trading_loop_v0_1.types import NoTradeReasonCode, SnapshotV01


class DummyProvider:
    def call(self, messages, *, model, json_schema=None, tool_schema=None) -> str:
        raise LLMRetryableError("timeout")


def _snapshot() -> SnapshotV01:
    return SnapshotV01(
        timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc),
        underlying="BTC",
        window_hours=24.0,
        spot_price=50000.0,
        has_open_short_call=False,
        open_short_call_count=0,
        open_short_call_symbols=[],
    )


def test_arbiter_returns_model_unavailable_on_router_failure() -> None:
    router = LLMRouter(
        model_chains={"trading_arbiter": ["p1:model-a"]},
        providers={"p1": DummyProvider()},
    )
    decision = arbiter_decide(_snapshot(), None, None, router=router)
    assert decision.decision == "NO_TRADE"
    assert decision.reason_code == NoTradeReasonCode.MODEL_UNAVAILABLE
