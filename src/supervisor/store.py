"""Job history storage for PR Supervisor with write safety."""

import asyncio
import json
import logging
import os
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .models import SupervisorJob

logger = logging.getLogger(__name__)

_store_lock = asyncio.Lock()
_approval_lock = threading.Lock()


@dataclass
class PRApprovalState:
    """Per-PR approval and pause state."""
    repo: str
    pr_number: int
    approved_by_telegram: bool = False
    approved_at: Optional[str] = None
    approved_by_user_id: Optional[int] = None
    paused: bool = False
    paused_at: Optional[str] = None
    paused_by_user_id: Optional[int] = None


class JobStore:
    """JSONL-based job history store with message registry support and write safety."""
    
    def __init__(self, storage_path: str = "/tmp/pr_supervisor_jobs/job_history.jsonl"):
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._jobs_cache: dict[str, SupervisorJob] = {}
        self._message_registry: dict[str, int] = {}
        self._registry_path = self.storage_path.parent / "message_registry.json"
        self._load_message_registry()
    
    def _load_cache(self) -> None:
        """Load jobs from JSONL file into cache."""
        if not self.storage_path.exists():
            return
        
        try:
            with open(self.storage_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data = json.loads(line)
                        job = SupervisorJob(**data)
                        self._jobs_cache[job.job_id] = job
        except Exception:
            pass
    
    def _save_job_sync(self, job: SupervisorJob) -> None:
        """Synchronously append or update job in storage (called within lock)."""
        self._jobs_cache[job.job_id] = job
        self._rewrite_store()
    
    def _rewrite_store(self) -> None:
        """Rewrite the entire JSONL store from cache with atomic flush."""
        try:
            temp_path = self.storage_path.with_suffix(".jsonl.tmp")
            with open(temp_path, "w") as f:
                for job in self._jobs_cache.values():
                    f.write(job.model_dump_json() + "\n")
                f.flush()
            temp_path.replace(self.storage_path)
        except Exception:
            pass
    
    def save(self, job: SupervisorJob) -> None:
        """Save a job to the store (thread-safe)."""
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(self._async_save(job))
        else:
            self._save_job_sync(job)
    
    async def _async_save(self, job: SupervisorJob) -> None:
        """Async save with lock protection."""
        async with _store_lock:
            self._save_job_sync(job)
    
    async def save_async(self, job: SupervisorJob) -> None:
        """Explicitly async save with lock protection."""
        async with _store_lock:
            self._save_job_sync(job)
    
    def get(self, job_id: str) -> Optional[SupervisorJob]:
        """Get a job by ID."""
        if not self._jobs_cache:
            self._load_cache()
        return self._jobs_cache.get(job_id)
    
    def get_by_sha(self, repo: str, pr_number: int, sha: str) -> Optional[SupervisorJob]:
        """Check if a job for this SHA already exists."""
        if not self._jobs_cache:
            self._load_cache()
        
        for job in self._jobs_cache.values():
            if (job.repo_full_name == repo and 
                job.pr_number == pr_number and 
                job.head_sha == sha):
                return job
        return None
    
    def get_run_count(self, repo: str, pr_number: int) -> int:
        """Get the number of runs for a PR (for run numbering)."""
        if not self._jobs_cache:
            self._load_cache()
        
        count = 0
        for job in self._jobs_cache.values():
            if job.repo_full_name == repo and job.pr_number == pr_number:
                count += 1
        return count
    
    def list_recent(self, limit: int = 50) -> list[SupervisorJob]:
        """List recent jobs."""
        if not self._jobs_cache:
            self._load_cache()
        
        jobs = sorted(
            self._jobs_cache.values(),
            key=lambda j: j.created_at,
            reverse=True
        )
        return jobs[:limit]
    
    def _load_message_registry(self) -> None:
        """Load Telegram message registry from JSON file."""
        if not self._registry_path.exists():
            return
        try:
            with open(self._registry_path, "r") as f:
                self._message_registry = json.load(f)
        except Exception:
            self._message_registry = {}
    
    def _save_message_registry(self) -> None:
        """Save Telegram message registry to JSON file with atomic write."""
        try:
            temp_path = self._registry_path.with_suffix(".json.tmp")
            with open(temp_path, "w") as f:
                json.dump(self._message_registry, f)
                f.flush()
            temp_path.replace(self._registry_path)
        except Exception:
            pass
    
    def get_telegram_message_id(self, repo: str, pr_number: int) -> Optional[int]:
        """Get stored Telegram message ID for a PR."""
        key = f"{repo}:{pr_number}"
        return self._message_registry.get(key)
    
    def set_telegram_message_id(self, repo: str, pr_number: int, message_id: int) -> None:
        """Store Telegram message ID for a PR."""
        key = f"{repo}:{pr_number}"
        self._message_registry[key] = message_id
        self._save_message_registry()
    
    def clear_telegram_message_id(self, repo: str, pr_number: int) -> None:
        """Clear Telegram message ID for a PR."""
        key = f"{repo}:{pr_number}"
        if key in self._message_registry:
            del self._message_registry[key]
            self._save_message_registry()
    
    def get_message_registry(self) -> dict[str, int]:
        """Get full message registry (for export)."""
        return self._message_registry.copy()
    
    def _get_approval_path(self) -> Path:
        """Get path to approval state file."""
        return self.storage_path.parent / "pr_approval_state.json"
    
    def _load_approval_state_unlocked(self) -> dict[str, dict]:
        """Load per-PR approval state from JSON file (caller must hold lock)."""
        path = self._get_approval_path()
        if not path.exists():
            return {}
        try:
            with open(path, "r") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse approval state file: {e}")
            return {}
        except OSError as e:
            logger.error(f"Failed to read approval state file: {e}")
            raise
    
    def _save_approval_state_unlocked(self, state: dict[str, dict]) -> None:
        """Save per-PR approval state with atomic write (caller must hold lock).
        
        Raises on failure - never silently ignores errors.
        """
        path = self._get_approval_path()
        temp_path = path.with_suffix(".json.tmp")
        try:
            with open(temp_path, "w") as f:
                json.dump(state, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(str(temp_path), str(path))
        except OSError as e:
            logger.error(f"Failed to save approval state: {e}")
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass
            raise
    
    def _load_approval_state(self) -> dict[str, dict]:
        """Load per-PR approval state from JSON file (thread-safe)."""
        with _approval_lock:
            return self._load_approval_state_unlocked()
    
    def _save_approval_state(self, state: dict[str, dict]) -> None:
        """Save per-PR approval state to JSON file (thread-safe)."""
        with _approval_lock:
            self._save_approval_state_unlocked(state)
    
    def get_pr_approval(self, repo: str, pr_number: int) -> PRApprovalState:
        """Get approval state for a PR."""
        state = self._load_approval_state()
        key = f"{repo}:{pr_number}"
        if key in state:
            data = state[key]
            return PRApprovalState(
                repo=repo,
                pr_number=pr_number,
                approved_by_telegram=data.get("approved_by_telegram", False),
                approved_at=data.get("approved_at"),
                approved_by_user_id=data.get("approved_by_user_id"),
                paused=data.get("paused", False),
                paused_at=data.get("paused_at"),
                paused_by_user_id=data.get("paused_by_user_id"),
            )
        return PRApprovalState(repo=repo, pr_number=pr_number)
    
    def set_pr_approval(
        self,
        repo: str,
        pr_number: int,
        approved: bool,
        user_id: Optional[int] = None,
    ) -> PRApprovalState:
        """Set Telegram approval for a PR.
        
        Uses timezone-aware UTC timestamps for audit trail.
        """
        with _approval_lock:
            state = self._load_approval_state_unlocked()
            key = f"{repo}:{pr_number}"
            
            if key not in state:
                state[key] = {"repo": repo, "pr_number": pr_number}
            
            state[key]["approved_by_telegram"] = approved
            if approved:
                state[key]["approved_at"] = datetime.now(timezone.utc).isoformat()
                state[key]["approved_by_user_id"] = user_id
            else:
                state[key]["approved_at"] = None
                state[key]["approved_by_user_id"] = None
            
            self._save_approval_state_unlocked(state)
        return self.get_pr_approval(repo, pr_number)
    
    def set_pr_paused(
        self,
        repo: str,
        pr_number: int,
        paused: bool,
        user_id: Optional[int] = None,
    ) -> PRApprovalState:
        """Set paused state for a PR.
        
        Uses timezone-aware UTC timestamps for audit trail.
        """
        with _approval_lock:
            state = self._load_approval_state_unlocked()
            key = f"{repo}:{pr_number}"
            
            if key not in state:
                state[key] = {"repo": repo, "pr_number": pr_number}
            
            state[key]["paused"] = paused
            if paused:
                state[key]["paused_at"] = datetime.now(timezone.utc).isoformat()
                state[key]["paused_by_user_id"] = user_id
            else:
                state[key]["paused_at"] = None
                state[key]["paused_by_user_id"] = None
            
            self._save_approval_state_unlocked(state)
        return self.get_pr_approval(repo, pr_number)
    
    def get_jobs_for_pr(self, repo: str, pr_number: int) -> list[SupervisorJob]:
        """Get all jobs for a specific PR."""
        if not self._jobs_cache:
            self._load_cache()
        
        jobs = [
            job for job in self._jobs_cache.values()
            if job.repo_full_name == repo and job.pr_number == pr_number
        ]
        return sorted(jobs, key=lambda j: j.created_at, reverse=True)
    
    def get_latest_job_for_pr(self, repo: str, pr_number: int) -> Optional[SupervisorJob]:
        """Get the latest job for a PR."""
        jobs = self.get_jobs_for_pr(repo, pr_number)
        return jobs[0] if jobs else None
