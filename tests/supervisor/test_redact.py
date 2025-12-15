"""Tests for secret redaction."""

import pytest

from src.supervisor.redact import redact_secrets, redact_job_for_api, REDACTED


class MockSettings:
    """Mock settings for testing."""
    
    def __init__(self):
        self.github_token = "ghp_abcdefghijklmnopqrstuvwxyz123456"
        self.github_webhook_secret = "my_webhook_secret_12345"
        self.openai_api_key = "sk-abcdefghij1234567890abcdefghij"
        self.telegram_bot_token = "1234567890:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefgh"
        self.gemini_api_key = "AIzaSyB1234567890abcdefghijklmnopqrstuv"


class TestRedactSecrets:
    """Tests for redact_secrets function."""
    
    def test_redacts_github_token(self):
        """Test redaction of GitHub personal access token."""
        settings = MockSettings()
        text = f"Token is {settings.github_token} here"
        result = redact_secrets(text, settings)
        
        assert settings.github_token not in result
        assert REDACTED in result
    
    def test_redacts_openai_key(self):
        """Test redaction of OpenAI API key."""
        settings = MockSettings()
        text = f"API key: {settings.openai_api_key}"
        result = redact_secrets(text, settings)
        
        assert settings.openai_api_key not in result
        assert REDACTED in result
    
    def test_redacts_telegram_token(self):
        """Test redaction of Telegram bot token."""
        settings = MockSettings()
        text = f"Bot token: {settings.telegram_bot_token}"
        result = redact_secrets(text, settings)
        
        assert settings.telegram_bot_token not in result
        assert REDACTED in result
    
    def test_redacts_webhook_secret(self):
        """Test redaction of webhook secret."""
        settings = MockSettings()
        text = f"Secret: {settings.github_webhook_secret}"
        result = redact_secrets(text, settings)
        
        assert settings.github_webhook_secret not in result
        assert REDACTED in result
    
    def test_redacts_github_pat_pattern(self):
        """Test redaction of GitHub PAT by regex pattern."""
        settings = MockSettings()
        settings.github_token = None
        
        text = "Found token ghp_1234567890abcdefghijklmnopqr in output"
        result = redact_secrets(text, settings)
        
        assert "ghp_" not in result or REDACTED in result
    
    def test_redacts_bearer_token(self):
        """Test redaction of Bearer tokens."""
        settings = MockSettings()
        
        text = "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWI"
        result = redact_secrets(text, settings)
        
        assert REDACTED in result
    
    def test_redacts_openai_pattern(self):
        """Test redaction of sk-* patterns."""
        settings = MockSettings()
        settings.openai_api_key = None
        
        text = "Using key sk-proj-abcdef1234567890abcdef12345678901234"
        result = redact_secrets(text, settings)
        
        assert REDACTED in result
    
    def test_empty_text(self):
        """Test handling of empty text."""
        settings = MockSettings()
        assert redact_secrets("", settings) == ""
        assert redact_secrets(None, settings) is None
    
    def test_no_secrets(self):
        """Test text with no secrets."""
        settings = MockSettings()
        text = "This is normal text without any secrets"
        result = redact_secrets(text, settings)
        
        assert result == text
        assert REDACTED not in result


class TestRedactJobForApi:
    """Tests for redact_job_for_api function."""
    
    def test_redacts_error_message(self):
        """Test redaction in error_message field."""
        settings = MockSettings()
        job_dict = {
            "error_message": f"Failed with token {settings.github_token}",
            "status": "error",
        }
        
        result = redact_job_for_api(job_dict, settings)
        
        assert settings.github_token not in result["error_message"]
        assert REDACTED in result["error_message"]
    
    def test_redacts_verification_failure(self):
        """Test redaction in verification failure_summary."""
        settings = MockSettings()
        job_dict = {
            "verification": {
                "failure_summary": f"Error: {settings.openai_api_key}",
                "checks": [],
            }
        }
        
        result = redact_job_for_api(job_dict, settings)
        
        assert settings.openai_api_key not in result["verification"]["failure_summary"]
    
    def test_redacts_check_output(self):
        """Test redaction in check stdout/stderr."""
        settings = MockSettings()
        job_dict = {
            "verification": {
                "checks": [
                    {
                        "command": "pytest",
                        "stdout": f"Token: {settings.telegram_bot_token}",
                        "stderr": "error output",
                    }
                ]
            }
        }
        
        result = redact_job_for_api(job_dict, settings)
        
        assert settings.telegram_bot_token not in result["verification"]["checks"][0]["stdout"]
    
    def test_redacts_codex_output(self):
        """Test redaction in fix_attempts codex_output."""
        settings = MockSettings()
        job_dict = {
            "fix_attempts": [
                {
                    "codex_output": f"Applied fix with {settings.gemini_api_key}",
                    "codex_prompt": "fix the issue",
                }
            ]
        }
        
        result = redact_job_for_api(job_dict, settings)
        
        assert settings.gemini_api_key not in result["fix_attempts"][0]["codex_output"]
    
    def test_preserves_non_sensitive_fields(self):
        """Test that non-sensitive fields are preserved."""
        settings = MockSettings()
        job_dict = {
            "job_id": "pr-123-abc",
            "status": "completed",
            "pr_number": 123,
        }
        
        result = redact_job_for_api(job_dict, settings)
        
        assert result["job_id"] == "pr-123-abc"
        assert result["status"] == "completed"
        assert result["pr_number"] == 123
