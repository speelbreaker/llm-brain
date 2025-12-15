"""Data models for PR Supervisor."""

from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    """Status of a supervisor job."""
    PENDING = "pending"
    RUNNING = "running"
    CHECKS_PASSED = "checks_passed"
    CHECKS_FAILED = "checks_failed"
    DEBATING = "debating"
    FIXING = "fixing"
    FIXED = "fixed"
    NEEDS_HUMAN = "needs_human"
    ERROR = "error"
    SKIPPED = "skipped"


class CheckResult(BaseModel):
    """Result of a single check command."""
    command: str
    exit_code: int
    passed: bool
    stdout: str = ""
    stderr: str = ""
    duration_seconds: float = 0.0
    truncated: bool = False


class VerificationReport(BaseModel):
    """Aggregated verification results."""
    commit_sha: str
    checks: list[CheckResult] = Field(default_factory=list)
    all_passed: bool = False
    failure_summary: str = ""
    failing_tests: list[str] = Field(default_factory=list)


class ArbiterDecision(BaseModel):
    """Decision from the Arbiter agent."""
    auto_fix_allowed: bool
    fix_objectives: list[str] = Field(default_factory=list)
    risk_level: str = "unknown"
    stop_reason: Optional[str] = None
    optimist_summary: str = ""
    skeptic_summary: str = ""
    arbiter_reasoning: str = ""


class DiffStats(BaseModel):
    """Statistics about code changes."""
    files_changed: int = 0
    lines_added: int = 0
    lines_removed: int = 0
    total_loc_changed: int = 0
    
    def within_thresholds(self, max_files: int, max_loc: int) -> bool:
        """Check if diff is within acceptable thresholds."""
        return self.files_changed <= max_files and self.total_loc_changed <= max_loc


class FixAttempt(BaseModel):
    """Record of a single fix attempt."""
    loop_number: int
    codex_prompt: str = ""
    codex_output: str = ""
    diff_stats: Optional[DiffStats] = None
    verification: Optional[VerificationReport] = None
    committed: bool = False
    commit_sha: Optional[str] = None


class SupervisorJob(BaseModel):
    """A supervisor job for a PR."""
    job_id: str
    repo_full_name: str
    pr_number: int
    head_sha: str
    head_ref: str
    base_ref: str
    pr_url: str
    is_fork: bool = False
    
    status: JobStatus = JobStatus.PENDING
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    workspace_path: Optional[str] = None
    verification: Optional[VerificationReport] = None
    arbiter_decision: Optional[ArbiterDecision] = None
    fix_attempts: list[FixAttempt] = Field(default_factory=list)
    
    final_message: str = ""
    error_message: Optional[str] = None
    
    def update_status(self, status: JobStatus) -> None:
        """Update job status and timestamp."""
        self.status = status
        self.updated_at = datetime.utcnow()


class WebhookPayload(BaseModel):
    """Parsed GitHub webhook payload."""
    action: str
    repo_full_name: str
    pr_number: int
    head_sha: str
    head_ref: str
    base_ref: str
    pr_url: str
    is_fork: bool
    sender: str
