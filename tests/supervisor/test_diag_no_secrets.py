import json
import re

import pytest
from httpx import AsyncClient, ASGITransport

from src.supervisor.app import app


FORBIDDEN_SUBSTRINGS = [
    "openai_api_key",
    "github_token",
    "webhook_secret",
    "token",
    "secret",
    "password",
    "private_key",
]
FORBIDDEN_REGEX = re.compile(r"(key|token|secret|password)", re.IGNORECASE)


@pytest.mark.asyncio
async def test_diag_endpoint_has_no_secrets():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get("/api/diag")
    assert resp.status_code == 200

    payload = resp.json()
    text = json.dumps(payload).lower()

    for bad in FORBIDDEN_SUBSTRINGS:
        assert bad not in text

    assert not FORBIDDEN_REGEX.search(text)
