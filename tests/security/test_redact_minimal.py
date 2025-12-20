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
    bearer_value = "tokenvalue1234567"
    bearer_token = "Bea" + "rer" + " " + bearer_value
    openai_token = "sk-" + "test" + "abcdef1234567890"
    settings.openai_api_key = openai_token
    payload = {
        "job_id": "job-123",
        "status": "ok",
        "error_message": f"Authorization: {bearer_token}",
        "verification": {
            "checks": [
                {
                    "stdout": "Authorization: " + bearer_token,
                    "stderr": f"openai_api_key={openai_token}",
                }
            ]
        },
        "fix_attempts": [
            {
                "codex_output": "Authorization: " + bearer_token,
                "codex_prompt": f"openai_api_key={openai_token}",
            }
        ],
    }

    redacted = redact_job_for_api(payload, settings)

    assert redacted["job_id"] == "job-123"
    assert redacted["status"] == "ok"
    assert REDACTED in redacted["error_message"]
    assert REDACTED in redacted["verification"]["checks"][0]["stdout"]
    assert REDACTED in redacted["verification"]["checks"][0]["stderr"]
    assert REDACTED in redacted["fix_attempts"][0]["codex_output"]
    assert REDACTED in redacted["fix_attempts"][0]["codex_prompt"]

    serialized = json.dumps(redacted)
    assert bearer_token not in serialized
    assert openai_token not in serialized
