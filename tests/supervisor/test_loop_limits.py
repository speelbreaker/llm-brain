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
    
    assert len(job.stage_history) == 0
    
    job.update_status(JobStatus.DEBATE)
    assert len(job.stage_history) == 1
    assert job.stage_history[0].stage == JobStatus.DEBATE
    assert isinstance(job.stage_history[0].timestamp, datetime)
    
    job.update_status(JobStatus.FIX_LINT)
    assert len(job.stage_history) == 2
    assert job.stage_history[1].stage == JobStatus.FIX_LINT
    
    # Ensure history is preserved
    assert job.stage_history[0].stage == JobStatus.DEBATE

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
