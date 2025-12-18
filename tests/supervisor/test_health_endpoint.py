import pytest
from httpx import AsyncClient, ASGITransport

from src.supervisor.app import app


@pytest.mark.asyncio
async def test_health_endpoint_shape_and_types():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get("/health")
    assert resp.status_code == 200

    data = resp.json()
    for field in ("ok", "enabled", "ready", "version"):
        assert field in data

    assert isinstance(data["ok"], bool)
    assert isinstance(data["enabled"], bool)
    assert isinstance(data["ready"], bool)
    assert isinstance(data["version"], str)
