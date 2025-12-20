"""Minimal redaction tests that avoid FastAPI/app imports."""

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
    payload = {
        "job_id": "job-123",
        "status": "ok",
        "metadata": {
            "auth_token": "ghp_ABCDEFGH1234567890XYZ",
            "note": "safe text",
        },
        "headers": "Authorization: Bearer abcdefghijklmnop",
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
    assert REDACTED in redacted["headers"]
    assert redacted["details"][0]["api_key"] == REDACTED
    assert REDACTED in redacted["details"][1]
    assert redacted["details"][2] == "normal string"
