"""Integration tests for Webhook Security (T008)."""

import pytest
import hmac
import hashlib
from fastapi.testclient import TestClient
from src.supervisor.app import app as supervisor_app
from src.supervisor.config import SupervisorSettings

@pytest.mark.anyio
async def test_webhook_signature_required(tmp_path):
    """Verify that webhooks require a valid signature if secret is configured."""
    settings = SupervisorSettings()
    settings.enabled = True
    settings.github_webhook_secret = "super_secret"
    settings.base_jobs_dir = str(tmp_path)
    
    supervisor_app.state.settings = settings
    supervisor_app.state.ready = True
    
    with TestClient(supervisor_app) as client:
        client.app.state.settings = settings
        client.app.state.ready = True
        
        # 1. No header -> 401
        response = client.post("/github/webhook", content=b"{}")
        assert response.status_code == 401
        assert response.json()["error"] == "invalid_signature"
        
        # 2. Wrong signature -> 401
        headers = {"X-Hub-Signature-256": "sha256=wrong"}
        response = client.post("/github/webhook", content=b"{}", headers=headers)
        assert response.status_code == 401
        
        # 3. Correct signature -> Not 401 (might be 400 if body is invalid for parser, but auth passed)
        body = b'{"action": "opened"}'
        computed_hash = hmac.new(
            settings.github_webhook_secret.encode("utf-8"),
            body,
            hashlib.sha256
        ).hexdigest()
        headers = {
            "X-Hub-Signature-256": f"sha256={computed_hash}",
            "X-GitHub-Event": "pull_request"
        }
        response = client.post("/github/webhook", content=body, headers=headers)
        # Auth passed, now it fails on PR parsing or other logic, but status is not 401
        assert response.status_code != 401
