import pytest
from httpx import AsyncClient, ASGITransport

from src.supervisor.app import app


@pytest.mark.asyncio
async def test_diag_endpoint_returns_expected_shape():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        r = await ac.get("/api/diag")
        assert r.status_code == 200
        data = r.json()
        assert data["ok"] is True
        assert "build_id" in data
        assert "code_paths" in data
        assert "models" in data
        assert "env_flags" in data
        assert "jobs" in data
        assert "notes" in data
        assert "workspace_py" in data["code_paths"]
        jobs = data["jobs"]
        assert "queue_depth" in jobs
        assert "worker_alive" in jobs
        assert "error_counts" in jobs
