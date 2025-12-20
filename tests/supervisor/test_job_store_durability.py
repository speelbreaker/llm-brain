"""Tests for job store durability across reloads."""

from datetime import datetime
from pathlib import Path

from src.supervisor.models import FixAttempt, JobStatus, SupervisorJob
from src.supervisor.store import JobStore


def test_fix_attempts_persist_across_reload(tmp_path: Path):
    store_path = tmp_path / "job_history.jsonl"
    store = JobStore(str(store_path))

    job = SupervisorJob(
        job_id="job-1",
        repo_full_name="owner/repo",
        pr_number=1,
        head_sha="abc",
        head_ref="feature",
        base_ref="main",
        pr_url="https://example.com/pr/1",
        status=JobStatus.FIXING,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )
    job.fix_attempts.append(
        FixAttempt(loop_number=1, committed=True, commit_sha="deadbeef")
    )
    store.save(job)

    reloaded = JobStore(str(store_path))
    loaded_job = reloaded.get("job-1")

    assert loaded_job is not None
    assert len(loaded_job.fix_attempts) == 1
    assert loaded_job.fix_attempts[0].commit_sha == "deadbeef"
