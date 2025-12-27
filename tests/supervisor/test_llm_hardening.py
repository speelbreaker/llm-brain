"""Tests for LLM hardening (T001-T004)."""

import pytest
from fastapi.testclient import TestClient
from src.supervisor.app import app as supervisor_app
from src.supervisor.config import SupervisorSettings

def test_diag_llm_available_flags(tmp_path):
    """Verify llm_available and codex_available flags in /api/diag."""
    settings = SupervisorSettings()
    settings.base_jobs_dir = str(tmp_path)
    settings.enabled = True
    
    # Case 1: All off
    settings.enable_codex = False
    settings.openai_api_key = None
    settings.gemini_api_key = None
    
    supervisor_app.state.settings = settings
    supervisor_app.state.ready = True
    supervisor_app.state.startup_errors = []

    with TestClient(supervisor_app) as client:
        supervisor_app.state.settings = settings
        response = client.get("/api/diag")
        assert response.status_code == 200
        payload = response.json()
        assert payload["llm_available"] is False
        assert payload["codex_available"] is False

        # Case 2: OpenAI Key present -> llm_available true
        settings.openai_api_key = "sk-test"
        settings.optimist_provider = "openai"
        settings.skeptic_provider = "openai"
        settings.arbiter_provider = "openai"
        supervisor_app.state.settings = settings
        response = client.get("/api/diag")
        payload = response.json()
        assert payload["llm_available"] is True
        assert payload["codex_available"] is False # codex still off

        # Case 3: Enable Codex + valid key + valid bin (mock)
        settings.enable_codex = True
        settings.codex_bin = "ls" # use 'ls' as a dummy executable binary
        supervisor_app.state.settings = settings
        response = client.get("/api/diag")
        payload = response.json()
        assert payload["llm_available"] is True
        assert payload["codex_available"] is True

        # Case 4: Gemini provider requested but no key
        settings.optimist_provider = "gemini"
        settings.skeptic_provider = "gemini"
        settings.arbiter_provider = "gemini"
        settings.gemini_api_key = None
        supervisor_app.state.settings = settings
        response = client.get("/api/diag")
        payload = response.json()
        assert payload["llm_available"] is False
        assert payload["codex_available"] is False
