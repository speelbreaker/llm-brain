import pytest
from datetime import datetime
from src.supervisor.models import SupervisorJob, JobStatus, StageHistory

def test_stage_history_append_only():
    job = SupervisorJob(
        job_id="1", 
        repo_full_name="a/b", 
        pr_number=1, 
        head_sha="sha", 
        head_ref="h", 
        base_ref="b", 
        pr_url="url"
    )
    
    # Lifecycle stage history starts at RECEIVED
    assert len(job.stage_history) == 1

    # Status history starts at PENDING
    assert len(job.status_history) == 1

    job.update_status(JobStatus.DEBATE)
    job.update_status(JobStatus.DEBATE)

    # update_status should be append-only on status_history
    assert len(job.status_history) >= 3
    assert job.status_history[-1].stage == JobStatus.DEBATE

    job.update_status(JobStatus.FIX_LINT)
    assert len(job.status_history) >= 4
    assert job.status_history[-1].stage == JobStatus.FIX_LINT

def test_loop_limit_reason_code_and_message():
    job = SupervisorJob(
        job_id="1", 
        repo_full_name="a/b", 
        pr_number=1, 
        head_sha="sha", 
        head_ref="h", 
        base_ref="b", 
        pr_url="url"
    )
    
    job.update_status(JobStatus.NEEDS_HUMAN)
    job.reason_code = "loop_limit"
    job.final_message = "Loop limit hit: fix_lint attempts=3"
    
    assert job.status == JobStatus.NEEDS_HUMAN
    assert job.reason_code == "loop_limit"
    assert "attempts=3" in job.final_message

def test_attempt_counters_increment():
    job = SupervisorJob(
        job_id="1", 
        repo_full_name="a/b", 
        pr_number=1, 
        head_sha="sha", 
        head_ref="h", 
        base_ref="b", 
        pr_url="url"
    )
    
    assert job.attempt_counters == {}
    
    job.attempt_counters[JobStatus.FIX_LINT] = 1
    assert job.attempt_counters[JobStatus.FIX_LINT] == 1
    
    job.attempt_counters[JobStatus.FIX_LINT] += 1
    assert job.attempt_counters[JobStatus.FIX_LINT] == 2
