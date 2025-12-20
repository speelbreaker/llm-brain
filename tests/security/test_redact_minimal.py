"""Minimal redaction tests that avoid FastAPI/app imports."""

import json

from src.supervisor.redact import REDACTED, redact_job_for_api


class MinimalSettings:
    """Settings stub for redaction tests."""

    def __init__(self):
        self.github_token = None
        self.github_webhook_secret = None
        self.openai_api_key = None
        self.telegram_bot_token = None
        self.gemini_api_key = None


def test_redacts_nested_tokens_and_preserves_safe_fields():
    settings = MinimalSettings()
    bearer_token = "Bea" + "rer" + " " + "testtoken123"
    payload = {
        "job_id": "job-123",
        "status": "ok",
        "metadata": {
            "auth_token": "ghp_ABCDEFGH1234567890XYZ",
            "note": "safe text",
        },
        "headers": {
            "Authorization": bearer_token,
            "raw": "Authorization: " + "Bea" + "rer" + " " + "abcdefghijklmnop",
        },
        "config": {"openai_api_key": "sk-test-abcdef"},
        "details": [
            {"api_key": "sk-test1234567890abcdef"},
            "GITHUB_TOKEN=ghp_1234567890ABCDEFGHIJKLMNOP",
            "normal string",
        ],
    }

    redacted = redact_job_for_api(payload, settings)

    assert redacted["job_id"] == "job-123"
    assert redacted["status"] == "ok"
    assert redacted["metadata"]["note"] == "safe text"
    assert redacted["metadata"]["auth_token"] == REDACTED
    assert redacted["headers"]["Authorization"] == REDACTED
    assert REDACTED in redacted["headers"]["raw"]
    assert redacted["config"]["openai_api_key"] == REDACTED
    assert redacted["details"][0]["api_key"] == REDACTED
    assert REDACTED in redacted["details"][1]
    assert redacted["details"][2] == "normal string"

    serialized = json.dumps(redacted)
    assert bearer_token not in serialized
    assert "sk-test-abcdef" not in serialized
