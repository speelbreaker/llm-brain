"""Route-level tests for webhook endpoint authentication."""

import hashlib
import hmac
import json
from unittest.mock import AsyncMock, patch, MagicMock

import pytest
from fastapi.testclient import TestClient


def make_signature(payload: bytes, secret: str) -> str:
    """Create a valid HMAC SHA-256 signature."""
    return "sha256=" + hmac.new(
        secret.encode("utf-8"),
        payload,
        hashlib.sha256
    ).hexdigest()


def make_pr_payload(action: str = "opened") -> dict:
    """Create a minimal valid PR webhook payload."""
    return {
        "action": action,
        "pull_request": {
            "number": 42,
            "html_url": "https://github.com/owner/repo/pull/42",
            "head": {
                "sha": "abc123def456",
                "ref": "feature-branch",
                "repo": {"full_name": "owner/repo", "fork": False},
            },
            "base": {"ref": "main"},
        },
        "repository": {"full_name": "owner/repo"},
        "sender": {"login": "testuser"},
    }


class TestWebhookRoute:
    """Route-level tests for /github/webhook endpoint."""
    
    @pytest.fixture
    def configured_app(self):
        """Create app with valid configuration."""
        with patch.dict("os.environ", {
            "SUPERVISOR_ENABLED": "1",
            "GITHUB_WEBHOOK_SECRET": "test_secret",
            "GITHUB_TOKEN": "test_token",
        }):
            with patch("src.supervisor.app.get_settings") as mock_settings:
                settings = MagicMock()
                settings.enabled = True
                settings.github_webhook_secret = "test_secret"
                settings.github_token = "test_token"
                settings.allow_forks = False
                settings.base_jobs_dir = "/tmp/test_jobs"
                settings.telegram_enabled = False
                mock_settings.return_value = settings
                
                from src.supervisor.app import app
                
                app.state.ready = True
                app.state.startup_errors = []
                app.state.settings = settings
                app.state.store = MagicMock()
                app.state.store.get_by_sha = MagicMock(return_value=None)
                app.state.store.save = MagicMock()
                app.state.job_queue = MagicMock()
                app.state.job_queue.put = AsyncMock()
                
                yield app
    
    @pytest.fixture
    def misconfigured_app(self):
        """Create app without webhook secret."""
        with patch("src.supervisor.app.get_settings") as mock_settings:
            settings = MagicMock()
            settings.enabled = True
            settings.github_webhook_secret = ""
            settings.github_token = "test_token"
            settings.allow_forks = False
            settings.base_jobs_dir = "/tmp/test_jobs"
            mock_settings.return_value = settings
            
            from src.supervisor.app import app
            
            app.state.ready = False
            app.state.startup_errors = ["GITHUB_WEBHOOK_SECRET"]
            app.state.settings = settings
            
            yield app
    
    def test_missing_secret_returns_503(self, misconfigured_app):
        """Test that missing webhook secret returns 503 misconfigured."""
        client = TestClient(misconfigured_app, raise_server_exceptions=False)
        
        payload = json.dumps(make_pr_payload()).encode()
        signature = make_signature(payload, "")
        
        response = client.post(
            "/github/webhook",
            content=payload,
            headers={
                "X-Hub-Signature-256": signature,
                "X-GitHub-Event": "pull_request",
                "Content-Type": "application/json",
            }
        )
        
        assert response.status_code == 503
        data = response.json()
        assert data["ok"] is False
        assert "misconfigured" in data.get("error", "")
    
    def test_valid_signature_accepted(self, configured_app):
        """Test that valid signature is accepted and job is queued."""
        client = TestClient(configured_app, raise_server_exceptions=False)
        
        payload = json.dumps(make_pr_payload()).encode()
        signature = make_signature(payload, "test_secret")
        
        response = client.post(
            "/github/webhook",
            content=payload,
            headers={
                "X-Hub-Signature-256": signature,
                "X-GitHub-Event": "pull_request",
                "Content-Type": "application/json",
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "queued"
        assert "job_id" in data
    
    def test_missing_signature_returns_401(self, configured_app):
        """Test that missing signature header returns 401."""
        client = TestClient(configured_app, raise_server_exceptions=False)
        
        payload = json.dumps(make_pr_payload()).encode()
        
        response = client.post(
            "/github/webhook",
            content=payload,
            headers={
                "X-GitHub-Event": "pull_request",
                "Content-Type": "application/json",
            }
        )
        
        assert response.status_code == 401
        data = response.json()
        assert data["error"] == "invalid_signature"
        assert "Missing" in data.get("detail", "")
    
    def test_invalid_signature_returns_401(self, configured_app):
        """Test that invalid signature returns 401."""
        client = TestClient(configured_app, raise_server_exceptions=False)
        
        payload = json.dumps(make_pr_payload()).encode()
        wrong_signature = make_signature(payload, "wrong_secret")
        
        response = client.post(
            "/github/webhook",
            content=payload,
            headers={
                "X-Hub-Signature-256": wrong_signature,
                "X-GitHub-Event": "pull_request",
                "Content-Type": "application/json",
            }
        )
        
        assert response.status_code == 401
        data = response.json()
        assert data["error"] == "invalid_signature"
        assert "verification failed" in data.get("detail", "").lower()
    
    def test_non_pr_event_ignored(self, configured_app):
        """Test that non-pull_request events are ignored."""
        client = TestClient(configured_app, raise_server_exceptions=False)
        
        payload = json.dumps({"action": "push"}).encode()
        signature = make_signature(payload, "test_secret")
        
        response = client.post(
            "/github/webhook",
            content=payload,
            headers={
                "X-Hub-Signature-256": signature,
                "X-GitHub-Event": "push",
                "Content-Type": "application/json",
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ignored"
    
    def test_disabled_supervisor(self):
        """Test that disabled supervisor returns disabled status."""
        with patch("src.supervisor.app.get_settings") as mock_settings:
            settings = MagicMock()
            settings.enabled = False
            mock_settings.return_value = settings
            
            from src.supervisor.app import app
            app.state.settings = settings
            
            client = TestClient(app, raise_server_exceptions=False)
            
            response = client.post(
                "/github/webhook",
                content=b"{}",
                headers={"Content-Type": "application/json"}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "disabled"
