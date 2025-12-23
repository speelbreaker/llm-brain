"""Data models for PR Supervisor."""

from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, model_validator


class JobStatus(str, Enum):
    """Status of a supervisor job."""
    PENDING = "pending"
    RUNNING = "running"
    CHECKS_PASSED = "checks_passed"
    CHECKS_FAILED = "checks_failed"
    DEBATE = "debate"
    FIX_LINT = "fix_lint"
    FIX_FORMAT = "fix_format"
    FIX_IMPORT = "fix_import"
    FIX_TESTS = "fix_tests"
    VERIFY = "verify"
    FIXING = "fixing"  # Legacy/Generic
    FIXED = "fixed"
    NEEDS_HUMAN = "needs_human"
    ERROR = "error"
    SKIPPED = "skipped"


class JobStage(str, Enum):
    """Lifecycle stage of a supervisor job."""
    RECEIVED = "received"
    ANALYZING = "analyzing"
    DEBATING = "debating"
    BYPASSED = "bypassed"
    FIXING = "fixing"
    SKIPPED = "skipped"
    VERIFYING = "verifying"
    COMMENTING = "commenting"
    DONE = "done"


class JobStageEntry(BaseModel):
    """Recorded stage transition with timestamps."""
    stage: JobStage
    entered_at: datetime = Field(default_factory=datetime.utcnow)
    exited_at: Optional[datetime] = None


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
    decision: str = "deny"
    reason: str = ""
    fix_objectives: list[str] = Field(default_factory=list)
    risk_level: str = "unknown"
    stop_reason: Optional[str] = None
    allowed_to_modify: list[str] = Field(default_factory=list)
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
    fixer: str = ""
    notes: list[str] = Field(default_factory=list)
    diff_stats: Optional[DiffStats] = None
    verification: Optional[VerificationReport] = None
    committed: bool = False
    commit_sha: Optional[str] = None


class StageHistory(BaseModel):
    """Record of a stage transition."""
    stage: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    payload: Optional[dict] = None


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
    reason_code: Optional[str] = None
    stage_history: list[StageHistory] = Field(default_factory=list)
    attempt_counters: dict[str, int] = Field(default_factory=dict)
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    stage: JobStage = JobStage.RECEIVED
    stage_entered_at: datetime = Field(default_factory=datetime.utcnow)
    stage_history: list[JobStageEntry] = Field(default_factory=list)

    debate_attempts: int = 0
    fix_attempts: int = 0
    verify_attempts: int = 0
    
    workspace_path: Optional[str] = None
    verification: Optional[VerificationReport] = None
    arbiter_decision: Optional[ArbiterDecision] = None
    fix_plan: Optional[dict] = None
    skeptic_report: Optional[dict] = None
    loop_decision: Optional[dict] = None
    fix_attempt_history: list[FixAttempt] = Field(default_factory=list)
    pr_comment_id: Optional[int] = None
    
    final_message: str = ""
    error_message: Optional[str] = None
    reason_code: Optional[str] = None
    
    def update_status(self, status: JobStatus) -> None:
        """Update job status and timestamp."""
        self.status = status
        self.updated_at = datetime.utcnow()
        self.stage_history.append(StageHistory(stage=status))

    def transition_stage(self, stage: JobStage) -> None:
        """Move job to a new stage and record the transition."""
        if self.stage == stage:
            return
        now = datetime.utcnow()
        if self.stage_history:
            self.stage_history[-1].exited_at = now
        self.stage = stage
        self.stage_entered_at = now
        self.stage_history.append(JobStageEntry(stage=stage, entered_at=now))
        self.updated_at = now

    def increment_debate_attempt(self) -> None:
        """Increment the debate attempt counter."""
        self.debate_attempts += 1
        self.updated_at = datetime.utcnow()

    def increment_fix_attempt(self) -> None:
        """Increment the fix attempt counter."""
        self.fix_attempts += 1
        self.updated_at = datetime.utcnow()

    def increment_verify_attempt(self) -> None:
        """Increment the verify attempt counter."""
        self.verify_attempts += 1
        self.updated_at = datetime.utcnow()

    @model_validator(mode="after")
    def _ensure_stage_history(self) -> "SupervisorJob":
        """Ensure stage history is initialized for new jobs."""
        if not self.stage_history:
            self.stage_history.append(
                JobStageEntry(stage=self.stage, entered_at=self.stage_entered_at)
            )
        return self


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