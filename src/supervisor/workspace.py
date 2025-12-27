"""Workspace isolation using git worktree."""

import asyncio
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Optional

from .config import SupervisorSettings
from .models import DiffStats

logger = logging.getLogger(__name__)

ACTIVE_SENTINEL = ".supervisor_active"


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
        base_ref: str = "main",
        pr_number: Optional[int] = None,
    ) -> str:
        """Set up an isolated workspace for a job using git worktree.
        
        Always fetches the PR head commit before creating worktree to handle
        PR updates (synchronize events) where new commits need to be fetched.
        
        Args:
            job_id: Unique job identifier
            repo_url: Repository clone URL
            head_sha: Commit SHA to checkout
            head_ref: Branch name of the PR head
            base_ref: Base branch name (default: main)
            pr_number: PR number for fallback fetch refspec
        """
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        repo_name = repo_url.split("/")[-1].replace(".git", "")
        bare_repo = self.cache_dir / f"{repo_name}.git"
        
        if not bare_repo.exists():
            await self._run_git(
                ["git", "clone", "--bare", repo_url, str(bare_repo)],
                cwd=str(self.cache_dir)
            )
        
        await self._fetch_refs(bare_repo, head_ref, base_ref, pr_number)
        
        await self._verify_commit_exists(bare_repo, head_sha, head_ref)
        
        workspace_path = self.base_dir / job_id
        if workspace_path.exists():
            try:
                await self._run_git(
                    ["git", "worktree", "remove", str(workspace_path), "--force"],
                    cwd=str(bare_repo)
                )
            except RuntimeError:
                pass
            shutil.rmtree(workspace_path, ignore_errors=True)
        
        await self._run_git(
            ["git", "worktree", "add", "--force", "--detach", str(workspace_path), head_sha],
            cwd=str(bare_repo)
        )
        
        sentinel = workspace_path / ACTIVE_SENTINEL
        sentinel.touch()
        
        return str(workspace_path)
    
    async def _fetch_refs(
        self,
        bare_repo: Path,
        head_ref: str,
        base_ref: str,
        pr_number: Optional[int],
    ) -> None:
        """Fetch required refs from origin.
        
        Fetches:
        1. Prune stale refs
        2. Head branch ref
        3. Base branch ref
        4. PR ref as fallback (if pr_number provided)
        """
        await self._run_git(
            ["git", "fetch", "--prune", "origin"],
            cwd=str(bare_repo)
        )
        
        try:
            await self._run_git(
                ["git", "fetch", "origin", f"refs/heads/{head_ref}:refs/remotes/origin/{head_ref}"],
                cwd=str(bare_repo)
            )
        except RuntimeError:
            logger.debug(f"Could not fetch head ref {head_ref}, may be from a fork")
        
        try:
            await self._run_git(
                ["git", "fetch", "origin", f"refs/heads/{base_ref}:refs/remotes/origin/{base_ref}"],
                cwd=str(bare_repo)
            )
        except RuntimeError:
            logger.debug(f"Could not fetch base ref {base_ref}")
        
        if pr_number:
            try:
                await self._run_git(
                    ["git", "fetch", "origin", f"refs/pull/{pr_number}/head:refs/remotes/pull/{pr_number}/head"],
                    cwd=str(bare_repo)
                )
            except RuntimeError:
                logger.debug(f"Could not fetch PR ref for PR #{pr_number}")
    
    async def _verify_commit_exists(
        self,
        bare_repo: Path,
        head_sha: str,
        head_ref: str,
    ) -> None:
        """Verify that the commit exists in the local repo.
        
        Raises RuntimeError if commit is not found after fetch.
        """
        try:
            await self._run_git(
                ["git", "cat-file", "-e", f"{head_sha}^{{commit}}"],
                cwd=str(bare_repo)
            )
        except RuntimeError as e:
            raise RuntimeError(
                f"Commit {head_sha[:8]} not found after fetch. "
                f"Branch: {head_ref}. This may indicate a force-push or the commit "
                f"was from a fork that cannot be fetched. Original error: {e}"
            )
    
    async def cleanup_workspace(self, job_id: str, bare_repo_name: Optional[str] = None) -> None:
        """Clean up a workspace after job completion with retry logic."""
        workspace_path = self.base_dir / job_id
        
        # 1. Clear sentinel
        sentinel = workspace_path / ACTIVE_SENTINEL
        if sentinel.exists():
            try:
                sentinel.unlink()
            except OSError:
                pass
        
        # 2. Try git worktree remove with retries
        if bare_repo_name:
            bare_repo = self.cache_dir / f"{bare_repo_name}.git"
            if bare_repo.exists():
                for attempt in range(3):
                    try:
                        await self._run_git(
                            ["git", "worktree", "remove", str(workspace_path), "--force"],
                            cwd=str(bare_repo)
                        )
                        break
                    except Exception as e:
                        if attempt < 2:
                            logger.debug(f"Git worktree remove failed (attempt {attempt+1}): {e}. Retrying...")
                            await asyncio.sleep(1)
                        else:
                            logger.warning(f"Git worktree remove failed after 3 attempts: {e}")
        
        # 3. Try shutil.rmtree with retries
        if workspace_path.exists():
            for attempt in range(3):
                try:
                    shutil.rmtree(workspace_path, ignore_errors=False)
                    break
                except Exception as e:
                    if attempt < 2:
                        logger.debug(f"shutil.rmtree failed (attempt {attempt+1}): {e}. Retrying...")
                        await asyncio.sleep(1)
                    else:
                        logger.warning(f"Final rmtree failed: {e}. Trying ignore_errors=True")
                        shutil.rmtree(workspace_path, ignore_errors=True)
    
    def mark_workspace_inactive(self, workspace_path: str) -> None:
        """Remove the active sentinel from a workspace (for use in finally blocks)."""
        path = Path(workspace_path)
        sentinel = path / ACTIVE_SENTINEL
        if sentinel.exists():
            try:
                sentinel.unlink()
            except OSError:
                pass
    
    async def cleanup_old_workspaces(self, sentinel_ttl_hours: int = 2) -> int:
        """Remove workspaces older than TTL.
        
        Returns number of workspaces cleaned up.
        
        SAFETY: Workspaces with a fresh .active sentinel file are skipped.
        If the sentinel is stale (mtime older than sentinel_ttl_hours), 
        the workspace is considered orphaned and eligible for cleanup.
        
        Args:
            sentinel_ttl_hours: Max age for sentinel to be considered fresh (default 2h)
        """
        ttl_hours = self.settings.workspace_ttl_hours
        if ttl_hours <= 0:
            return 0
        
        if not self.base_dir.exists():
            return 0
        
        cutoff = time.time() - (ttl_hours * 3600)
        sentinel_cutoff = time.time() - (sentinel_ttl_hours * 3600)
        cleaned = 0
        
        for path in self.base_dir.iterdir():
            if path.name.startswith("_"):
                continue
            if not path.is_dir():
                continue
            
            sentinel = path / ACTIVE_SENTINEL
            if sentinel.exists():
                try:
                    sentinel_mtime = sentinel.stat().st_mtime
                    if sentinel_mtime > sentinel_cutoff:
                        logger.debug(f"Skipping active workspace (fresh sentinel): {path.name}")
                        continue
                    else:
                        logger.warning(f"Stale sentinel detected in {path.name}, treating as orphaned")
                except OSError:
                    pass
            
            try:
                mtime = path.stat().st_mtime
                if mtime < cutoff:
                    logger.info(f"Cleaning up old workspace: {path.name}")
                    shutil.rmtree(path, ignore_errors=True)
                    cleaned += 1
            except Exception as e:
                logger.warning(f"Failed to clean workspace {path.name}: {e}")
        
        if cleaned > 0:
            logger.info(f"Cleaned up {cleaned} old workspaces (TTL: {ttl_hours}h)")
        
        return cleaned
    
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
