"""Workspace isolation using git worktree."""

import asyncio
import os
import shutil
from pathlib import Path
from typing import Optional

from .config import SupervisorSettings
from .models import DiffStats


class WorkspaceManager:
    """Manages isolated workspaces for PR verification."""
    
    def __init__(self, settings: SupervisorSettings):
        self.settings = settings
        self.base_dir = Path(settings.base_jobs_dir)
        self.cache_dir = self.base_dir / "_cache"
    
    async def setup_workspace(
        self,
        job_id: str,
        repo_url: str,
        head_sha: str,
        head_ref: str,
    ) -> str:
        """Set up an isolated workspace for a job using git worktree."""
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        repo_name = repo_url.split("/")[-1].replace(".git", "")
        bare_repo = self.cache_dir / f"{repo_name}.git"
        
        if not bare_repo.exists():
            await self._run_git(
                ["git", "clone", "--bare", repo_url, str(bare_repo)],
                cwd=str(self.cache_dir)
            )
        else:
            await self._run_git(
                ["git", "fetch", "--all"],
                cwd=str(bare_repo)
            )
        
        workspace_path = self.base_dir / job_id
        if workspace_path.exists():
            shutil.rmtree(workspace_path)
        
        await self._run_git(
            ["git", "worktree", "add", str(workspace_path), head_sha],
            cwd=str(bare_repo)
        )
        
        return str(workspace_path)
    
    async def cleanup_workspace(self, job_id: str, bare_repo_name: Optional[str] = None) -> None:
        """Clean up a workspace after job completion."""
        workspace_path = self.base_dir / job_id
        
        if bare_repo_name:
            bare_repo = self.cache_dir / f"{bare_repo_name}.git"
            if bare_repo.exists():
                try:
                    await self._run_git(
                        ["git", "worktree", "remove", str(workspace_path), "--force"],
                        cwd=str(bare_repo)
                    )
                except Exception:
                    pass
        
        if workspace_path.exists():
            shutil.rmtree(workspace_path, ignore_errors=True)
    
    async def get_diff_stats(self, workspace_path: str) -> DiffStats:
        """Get diff statistics for uncommitted changes."""
        try:
            result = await self._run_git(
                ["git", "diff", "--stat", "--numstat"],
                cwd=workspace_path
            )
            
            lines = result.strip().split("\n") if result.strip() else []
            files_changed = 0
            lines_added = 0
            lines_removed = 0
            
            for line in lines:
                parts = line.split("\t")
                if len(parts) >= 3:
                    try:
                        added = int(parts[0]) if parts[0] != "-" else 0
                        removed = int(parts[1]) if parts[1] != "-" else 0
                        lines_added += added
                        lines_removed += removed
                        files_changed += 1
                    except ValueError:
                        continue
            
            return DiffStats(
                files_changed=files_changed,
                lines_added=lines_added,
                lines_removed=lines_removed,
                total_loc_changed=lines_added + lines_removed,
            )
        except Exception:
            return DiffStats()
    
    async def commit_and_push(
        self,
        workspace_path: str,
        message: str,
        branch: str,
    ) -> Optional[str]:
        """Commit changes and push to the PR branch."""
        await self._run_git(["git", "add", "-A"], cwd=workspace_path)
        
        status = await self._run_git(["git", "status", "--porcelain"], cwd=workspace_path)
        if not status.strip():
            return None
        
        await self._run_git(
            ["git", "commit", "-m", message],
            cwd=workspace_path
        )
        
        result = await self._run_git(
            ["git", "rev-parse", "HEAD"],
            cwd=workspace_path
        )
        commit_sha = result.strip()
        
        await self._run_git(
            ["git", "push", "origin", f"HEAD:{branch}"],
            cwd=workspace_path
        )
        
        return commit_sha
    
    async def _run_git(self, cmd: list[str], cwd: str) -> str:
        """Run a git command and return stdout."""
        env = os.environ.copy()
        env["GIT_TERMINAL_PROMPT"] = "0"
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            raise RuntimeError(
                f"Git command failed: {' '.join(cmd)}\n"
                f"stderr: {stderr.decode()}"
            )
        
        return stdout.decode()
