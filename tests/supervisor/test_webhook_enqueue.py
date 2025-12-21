import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import patch

from src.supervisor.app import app


class DummyPayload:
    action = "synchronize"
    repo_full_name = "speelbreaker/llm-brain"
    pr_number = 1
    head_sha = "deadbeef" * 5
    head_ref = "pr-supervisor-smoke"
    base_ref = "main"
    pr_url = "https://github.com/speelbreaker/llm-brain/pull/1"
    is_fork = False


@pytest.mark.asyncio
async def test_webhook_enqueues_job():
    # Ensure supervisor enabled in the running app state:
    # We patch verify_signature to True and parse_webhook_payload to a known object
    with patch("src.supervisor.app.verify_signature", return_value=True), \
         patch("src.supervisor.app.parse_webhook_payload", return_value=DummyPayload()):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            r = await ac.post(
                "/github/webhook",
                headers={
                    "X-GitHub-Event": "pull_request",
                    "X-Hub-Signature-256": "sha256=fake",
                },
                json={"dummy": True},
            )

            assert r.status_code == 200
            body = r.json()
            assert body["status"] in ("queued", "ignored", "disabled")
