"""Tests covering the supervisor diagnostics endpoint."""

from fastapi.testclient import TestClient

from src.supervisor.app import app as supervisor_app
from src.supervisor.config import SupervisorSettings


class DummyTask:
    def __init__(self, done_value: bool):
        self._done_value = done_value

    def done(self) -> bool:
        return self._done_value

    def cancel(self) -> None:
        pass

    def __await__(self):
        if False:
            yield
        return None


def test_diag_endpoint_reports_required_keys(tmp_path):
    settings = SupervisorSettings()
    settings.enabled = True
    settings.debug = True
    settings.autofix_push = True
    settings.autofix_dry_run = False
    settings.enable_codex = True
    settings.codex_bin = "/nonexistent-codex-diag"
    settings.github_token = "ghp_perm"
    settings.github_webhook_secret = "secret"
    settings.base_jobs_dir = str(tmp_path)

    supervisor_app.state.settings = settings
    supervisor_app.state.ready = True
    supervisor_app.state.startup_errors = []
    supervisor_app.state.use_preconfigured_settings = True
    supervisor_app.state.use_preconfigured_job_queue = True

    with TestClient(supervisor_app, raise_server_exceptions=False) as client:
        client.app.state.settings = settings
        client.app.state.ready = True
        client.app.state.startup_errors = []
        client.app.state.supervisor_worker_task = DummyTask(done_value=False)
        response = client.get("/api/diag")
        assert response.status_code == 200

        payload = response.json()
        assert payload["ok"] is True
        assert payload["worker_alive"] is True
        assert payload["debug_enabled"] is True
        assert payload["push_enabled"] is True
        assert payload["dry_run"] is False
        assert payload["codex_available"] is False
        assert "provider_health" in payload
        assert isinstance(payload["provider_health"], dict)
        assert "build_id" in payload
