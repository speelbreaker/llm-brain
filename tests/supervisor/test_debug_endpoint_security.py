import asyncio
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from src.supervisor.app import app
from src.supervisor.config import SupervisorSettings
from src.supervisor.store import JobStore


def _configure_app_state(settings: SupervisorSettings, store_path):
    app.state.settings = settings
    app.state.ready = True
    app.state.startup_errors = []
    app.state.job_queue = asyncio.Queue()
    app.state.store = JobStore(store_path)
    app.state.github_client = AsyncMock()
    app.state.github_client.get_pr_info.return_value = {
        "html_url": "https://example.com/pr/1",
        "title": "Test PR",
        "body": "",
        "labels": [{"name": "autofix-ok"}],
    }
    app.state.github_client.get_pr_files.return_value = []
    app.state.github_client.post_pr_comment.return_value = {}


def test_debug_disabled_returns_404(tmp_path):
    settings = SupervisorSettings()
    settings.debug_enabled = False
    settings.debug_token = "secret-token"
    settings.enabled = True

    with TestClient(app, raise_server_exceptions=False) as client:
        _configure_app_state(settings, tmp_path / "jobs.jsonl")
        resp = client.post(
            "/debug/simulate_pr_event",
            json={"repo": "owner/repo", "pr_number": 1},
        )
    assert resp.status_code == 404


def test_debug_requires_token(tmp_path):
    settings = SupervisorSettings()
    settings.debug_enabled = True
    settings.debug_token = "secret-token"
    settings.enabled = True

    with TestClient(app, raise_server_exceptions=False) as client:
        _configure_app_state(settings, tmp_path / "jobs.jsonl")
        resp = client.post(
            "/debug/simulate_pr_event",
            json={"repo": "owner/repo", "pr_number": 1},
        )
    assert resp.status_code == 401


def test_debug_rejects_non_localhost(monkeypatch, tmp_path):
    settings = SupervisorSettings()
    settings.debug_enabled = True
    settings.debug_token = "secret-token"
    settings.enabled = True

    monkeypatch.setattr(
        "src.supervisor.app._is_local_request", lambda request: False, raising=True
    )

    with TestClient(app, raise_server_exceptions=False) as client:
        _configure_app_state(settings, tmp_path / "jobs.jsonl")
        resp = client.post(
            "/debug/simulate_pr_event",
            json={"repo": "owner/repo", "pr_number": 1},
            headers={"X-Debug-Token": "secret-token"},
        )
    assert resp.status_code in (401, 403)


def test_debug_allows_local_with_token(tmp_path):
    settings = SupervisorSettings()
    settings.debug_enabled = True
    settings.debug_token = "secret-token"
    settings.enabled = True

    with TestClient(app, raise_server_exceptions=False) as client:
        _configure_app_state(settings, tmp_path / "jobs.jsonl")
        resp = client.post(
            "/debug/simulate_pr_event",
            json={"repo": "owner/repo", "pr_number": 1},
            headers={"X-Debug-Token": "secret-token"},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert data["status"] == "queued"
