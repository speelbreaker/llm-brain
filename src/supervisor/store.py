"""Job history storage for PR Supervisor with write safety."""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from .models import SupervisorJob

_store_lock = asyncio.Lock()


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
