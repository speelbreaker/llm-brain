"""Tests for GitHub retry behavior."""

from unittest.mock import AsyncMock

import httpx
import pytest

from src.supervisor.github import GitHubClient
from src.supervisor import retry as retry_module


@pytest.mark.asyncio
async def test_github_retries_on_429(monkeypatch):
    client = GitHubClient(token="dummy")

    req = httpx.Request("GET", "https://api.github.com/repos/owner/repo/pulls/1")
    resp_429 = httpx.Response(429, request=req)
    resp_200 = httpx.Response(200, request=req, json={"number": 1})

    fake_client = AsyncMock()
    fake_client.request = AsyncMock(side_effect=[resp_429, resp_200])
    monkeypatch.setattr(client, "_get_client", AsyncMock(return_value=fake_client))
    monkeypatch.setattr(retry_module.asyncio, "sleep", AsyncMock())

    data = await client.get_pr_info("owner/repo", 1)

    assert data["number"] == 1
    assert fake_client.request.call_count == 2
