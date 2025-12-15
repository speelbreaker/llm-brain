"""Route-level tests for Jobs API truncation."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.supervisor.app import MAX_TRUNCATE_CHARS


def make_mock_job(
    job_id: str = "test-job-123",
    huge_stdout: bool = False,
    huge_stderr: bool = False,
    huge_failure_summary: bool = False,
    huge_codex_output: bool = False,
):
    """Create a mock job with optionally huge fields."""
    import copy
    job = MagicMock()
    
    huge_text = "Error at line 1. " * 600
    
    job_dict = {
        "job_id": job_id,
        "status": "completed",
        "repo_full_name": "owner/repo",
        "pr_number": 42,
        "head_sha": "abc123def456",
        "verification": {
            "all_passed": False,
            "failure_summary": huge_text if huge_failure_summary else "short failure",
            "checks": [
                {
                    "command": "pytest",
                    "passed": False,
                    "stdout": huge_text if huge_stdout else "short stdout",
                    "stderr": huge_text if huge_stderr else "short stderr",
                }
            ],
        },
        "fix_attempts": [
            {
                "loop_number": 1,
                "codex_output": huge_text if huge_codex_output else "short output",
                "codex_prompt": "Fix the failing tests",
            }
        ],
    }
    
    job.model_dump.side_effect = lambda: copy.deepcopy(job_dict)
    return job, job_dict


class TestJobsApiTruncation:
    """Tests for /jobs API endpoint truncation."""
    
    @pytest.fixture
    def app_with_store(self):
        """Create app with mock store containing jobs."""
        with patch("src.supervisor.app.get_settings") as mock_settings:
            settings = MagicMock()
            settings.enabled = True
            settings.github_webhook_secret = "test_secret"
            settings.github_token = "test_token"
            settings.base_jobs_dir = "/tmp/test_jobs"
            mock_settings.return_value = settings
            
            from src.supervisor.app import app
            
            app.state.settings = settings
            app.state.ready = True
            app.state.startup_errors = []
            app.state.store = MagicMock()
            
            yield app
    
    def test_list_jobs_truncates_stdout(self, app_with_store):
        """Test that /jobs endpoint truncates large stdout."""
        job, _ = make_mock_job(huge_stdout=True)
        app_with_store.state.store.list_recent.return_value = [job]
        
        client = TestClient(app_with_store)
        response = client.get("/jobs")
        
        assert response.status_code == 200
        data = response.json()
        
        assert len(data["jobs"]) == 1
        check = data["jobs"][0]["verification"]["checks"][0]
        
        assert len(check["stdout"]) <= MAX_TRUNCATE_CHARS
        assert check["stdout_truncated"] is True
    
    def test_list_jobs_truncates_stderr(self, app_with_store):
        """Test that /jobs endpoint truncates large stderr."""
        job, _ = make_mock_job(huge_stderr=True)
        app_with_store.state.store.list_recent.return_value = [job]
        
        client = TestClient(app_with_store)
        response = client.get("/jobs")
        
        assert response.status_code == 200
        data = response.json()
        
        check = data["jobs"][0]["verification"]["checks"][0]
        
        assert len(check["stderr"]) <= MAX_TRUNCATE_CHARS
        assert check["stderr_truncated"] is True
    
    def test_list_jobs_truncates_failure_summary(self, app_with_store):
        """Test that /jobs endpoint truncates large failure_summary."""
        job, _ = make_mock_job(huge_failure_summary=True)
        app_with_store.state.store.list_recent.return_value = [job]
        
        client = TestClient(app_with_store)
        response = client.get("/jobs")
        
        assert response.status_code == 200
        data = response.json()
        
        verification = data["jobs"][0]["verification"]
        
        assert len(verification["failure_summary"]) <= MAX_TRUNCATE_CHARS
        assert verification["failure_summary_truncated"] is True
    
    def test_get_job_truncates_all_fields(self, app_with_store):
        """Test that /jobs/{id} endpoint truncates all large fields."""
        job, _ = make_mock_job(
            huge_stdout=True,
            huge_stderr=True,
            huge_failure_summary=True,
            huge_codex_output=True,
        )
        app_with_store.state.store.get.return_value = job
        
        client = TestClient(app_with_store)
        response = client.get("/jobs/test-job-123")
        
        assert response.status_code == 200
        data = response.json()
        
        verification = data["verification"]
        assert len(verification["failure_summary"]) <= MAX_TRUNCATE_CHARS
        assert verification["failure_summary_truncated"] is True
        
        check = verification["checks"][0]
        assert len(check["stdout"]) <= MAX_TRUNCATE_CHARS
        assert check["stdout_truncated"] is True
        assert len(check["stderr"]) <= MAX_TRUNCATE_CHARS
        assert check["stderr_truncated"] is True
        
        attempt = data["fix_attempts"][0]
        assert len(attempt["codex_output"]) <= MAX_TRUNCATE_CHARS
        assert attempt["codex_output_truncated"] is True
    
    def test_short_fields_not_truncated(self, app_with_store):
        """Test that short fields are not marked as truncated."""
        job, _ = make_mock_job()
        app_with_store.state.store.get.return_value = job
        
        client = TestClient(app_with_store)
        response = client.get("/jobs/test-job-123")
        
        assert response.status_code == 200
        data = response.json()
        
        check = data["verification"]["checks"][0]
        assert check["stdout"] == "short stdout"
        assert check["stdout_truncated"] is False
        assert check["stderr"] == "short stderr"
        assert check["stderr_truncated"] is False
    
    def test_job_not_found_returns_404(self, app_with_store):
        """Test that missing job returns 404."""
        app_with_store.state.store.get.return_value = None
        
        client = TestClient(app_with_store)
        response = client.get("/jobs/nonexistent")
        
        assert response.status_code == 404
