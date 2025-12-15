"""Job history storage for PR Supervisor."""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from .models import SupervisorJob


class JobStore:
    """JSONL-based job history store."""
    
    def __init__(self, storage_path: str = "/tmp/pr_supervisor_jobs/job_history.jsonl"):
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._jobs_cache: dict[str, SupervisorJob] = {}
    
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
    
    def _save_job(self, job: SupervisorJob) -> None:
        """Append or update job in storage."""
        self._jobs_cache[job.job_id] = job
        self._rewrite_store()
    
    def _rewrite_store(self) -> None:
        """Rewrite the entire JSONL store from cache."""
        try:
            with open(self.storage_path, "w") as f:
                for job in self._jobs_cache.values():
                    f.write(job.model_dump_json() + "\n")
        except Exception:
            pass
    
    def save(self, job: SupervisorJob) -> None:
        """Save a job to the store."""
        self._save_job(job)
    
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
