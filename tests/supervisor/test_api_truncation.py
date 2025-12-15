"""Tests for API payload truncation."""

import pytest

from src.supervisor.app import truncate_field, truncate_job_for_api, MAX_TRUNCATE_CHARS


class TestTruncateField:
    """Tests for truncate_field function."""
    
    def test_short_text_not_truncated(self):
        """Test that short text is not truncated."""
        result = truncate_field("short text")
        
        assert result["value"] == "short text"
        assert result["truncated"] is False
    
    def test_none_value_handled(self):
        """Test that None value is handled."""
        result = truncate_field(None)
        
        assert result["value"] is None
        assert result["truncated"] is False
    
    def test_empty_string_handled(self):
        """Test that empty string is handled."""
        result = truncate_field("")
        
        assert result["value"] is None
        assert result["truncated"] is False
    
    def test_long_text_truncated(self):
        """Test that long text is truncated."""
        long_text = "A" * 10000
        result = truncate_field(long_text, max_chars=5000)
        
        assert len(result["value"]) == 5000
        assert result["truncated"] is True
        assert result["original_length"] == 10000
        assert result["max_chars"] == 5000
    
    def test_exact_length_not_truncated(self):
        """Test that text at exact max length is not truncated."""
        text = "A" * 5000
        result = truncate_field(text, max_chars=5000)
        
        assert result["value"] == text
        assert result["truncated"] is False


class TestTruncateJobForApi:
    """Tests for truncate_job_for_api function."""
    
    def test_truncates_failure_summary(self):
        """Test that failure_summary is truncated."""
        job_dict = {
            "verification": {
                "failure_summary": "A" * 10000,
                "checks": [],
            }
        }
        
        result = truncate_job_for_api(job_dict)
        
        assert len(result["verification"]["failure_summary"]) == MAX_TRUNCATE_CHARS
        assert result["verification"]["failure_summary_truncated"] is True
    
    def test_truncates_check_stdout(self):
        """Test that check stdout is truncated."""
        job_dict = {
            "verification": {
                "checks": [
                    {
                        "command": "pytest",
                        "stdout": "X" * 10000,
                        "stderr": "",
                    }
                ]
            }
        }
        
        result = truncate_job_for_api(job_dict)
        
        check = result["verification"]["checks"][0]
        assert len(check["stdout"]) == MAX_TRUNCATE_CHARS
        assert check["stdout_truncated"] is True
    
    def test_truncates_check_stderr(self):
        """Test that check stderr is truncated."""
        job_dict = {
            "verification": {
                "checks": [
                    {
                        "command": "pytest",
                        "stdout": "",
                        "stderr": "E" * 10000,
                    }
                ]
            }
        }
        
        result = truncate_job_for_api(job_dict)
        
        check = result["verification"]["checks"][0]
        assert len(check["stderr"]) == MAX_TRUNCATE_CHARS
        assert check["stderr_truncated"] is True
    
    def test_truncates_codex_output(self):
        """Test that codex_output is truncated."""
        job_dict = {
            "fix_attempts": [
                {
                    "codex_output": "O" * 10000,
                    "codex_prompt": "Fix the issue",
                }
            ]
        }
        
        result = truncate_job_for_api(job_dict)
        
        attempt = result["fix_attempts"][0]
        assert len(attempt["codex_output"]) == MAX_TRUNCATE_CHARS
        assert attempt["codex_output_truncated"] is True
    
    def test_truncates_codex_prompt(self):
        """Test that codex_prompt is truncated."""
        job_dict = {
            "fix_attempts": [
                {
                    "codex_output": "Short output",
                    "codex_prompt": "P" * 10000,
                }
            ]
        }
        
        result = truncate_job_for_api(job_dict)
        
        attempt = result["fix_attempts"][0]
        assert len(attempt["codex_prompt"]) == MAX_TRUNCATE_CHARS
        assert attempt["codex_prompt_truncated"] is True
    
    def test_handles_missing_verification(self):
        """Test that missing verification field is handled."""
        job_dict = {"job_id": "test", "status": "running"}
        
        result = truncate_job_for_api(job_dict)
        
        assert result["job_id"] == "test"
        assert result["status"] == "running"
    
    def test_handles_missing_fix_attempts(self):
        """Test that missing fix_attempts field is handled."""
        job_dict = {"job_id": "test", "verification": {"failure_summary": "short"}}
        
        result = truncate_job_for_api(job_dict)
        
        assert result["job_id"] == "test"
    
    def test_preserves_short_fields(self):
        """Test that short fields are not modified."""
        job_dict = {
            "verification": {
                "failure_summary": "short summary",
                "checks": [
                    {"command": "pytest", "stdout": "short out", "stderr": "short err"}
                ],
            },
            "fix_attempts": [
                {"codex_output": "short", "codex_prompt": "short"}
            ],
        }
        
        result = truncate_job_for_api(job_dict)
        
        assert result["verification"]["failure_summary"] == "short summary"
        assert result["verification"]["failure_summary_truncated"] is False
        assert result["verification"]["checks"][0]["stdout"] == "short out"
        assert result["verification"]["checks"][0]["stdout_truncated"] is False
