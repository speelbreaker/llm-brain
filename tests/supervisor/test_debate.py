import pytest

from src.supervisor.config import SupervisorSettings
from src.supervisor import debate


class _FakeProvider:
    """Minimal fake provider that returns a payload containing a role field."""

    def __init__(self, role: str):
        self.role = role

    async def generate_json(self, **kwargs):
        return {
            "role": self.role,
            "summary": f"{self.role} summary",
            "bullets": ["point"],
            "auto_fix_allowed": True if self.role == "arbiter" else None,
            "objectives": ["do the thing"] if self.role == "arbiter" else None,
            "risk_level": "low" if self.role == "arbiter" else None,
            "stop_reason": None,
        }

    async def close(self):
        return None


@pytest.mark.asyncio
async def test_call_agent_strips_role_from_payload(monkeypatch):
    settings = SupervisorSettings()

    def _fake_get_provider(role, _settings):
        return _FakeProvider(role), "fake-model"

    monkeypatch.setattr(debate, "get_provider_for_role", _fake_get_provider)

    system = debate.DebateSystem(settings)
    result = await system._call_agent("optimist", "context")

    assert result["role"] == "optimist"
    assert result["summary"] == "optimist summary"


@pytest.mark.asyncio
async def test_call_arbiter_strips_role_from_payload(monkeypatch):
    settings = SupervisorSettings()

    def _fake_get_provider(role, _settings):
        return _FakeProvider(role), "fake-model"

    monkeypatch.setattr(debate, "get_provider_for_role", _fake_get_provider)

    system = debate.DebateSystem(settings)
    decision = await system._call_arbiter("context", {"summary": ""}, {"summary": ""})

    assert decision.auto_fix_allowed is True
    assert decision.risk_level == "low"
