"""Workspace isolation using git worktree.

Core rule: BEFORE we try `git worktree add <head_sha>`, we must ensure the
bare mirror has fetched refs that contain that SHA (PR branch / PR ref / base).
"""

from __future__ import annotations

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


def _safe_ref_name(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".", "/") else "_" for ch in (s or ""))


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
        pr_number: int | None = None,
    ) -> str:
        """Create a worktree checked out at head_sha.

        Fixes "fatal: invalid reference" by fetching:
          - PR ref (refs/pull/<n>/head) when pr_number is known
          - PR branch (refs/heads/<head_ref>)
          - base branch (refs/heads/<base_ref>)
        then verifying head_sha exists (git cat-file -e) before worktree add.
        """
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        repo_name = repo_url.split("/")[-1].replace(".git", "") or "repo"
        bare_repo = self.cache_dir / f"{repo_name}.git"

        # Ensure bare mirror exists
        if not bare_repo.exists():
            logger.info("Cloning bare repo to %s", bare_repo)
            await self._run_git(["git", "clone", "--bare", repo_url, str(bare_repo)], cwd=str(self.cache_dir))
        else:
            # Ensure origin URL is correct (in case repo_url changed)
            try:
                await self._run_git(["git", "remote", "set-url", "origin", repo_url], cwd=str(bare_repo))
            except Exception:
                pass

        # Always fetch base branch
        base_ref = base_ref or "main"
        base_ref = _safe_ref_name(base_ref)
        try:
            await self._run_git(
                ["git", "fetch", "--prune", "origin", f"+refs/heads/{base_ref}:refs/heads/{base_ref}"],
                cwd=str(bare_repo),
            )
        except Exception as e:
            logger.warning("Fetch base_ref %s failed: %s", base_ref, e)

        # Fetch PR ref (most reliable on GitHub) if we know the PR number
        if pr_number is not None:
            try:
                await self._run_git(
                    ["git", "fetch", "origin", f"+refs/pull/{pr_number}/head:refs/heads/pr/{pr_number}"],
                    cwd=str(bare_repo),
                )
            except Exception as e:
                logger.warning("Fetch PR ref pull/%s/head failed: %s", pr_number, e)

        # Fetch the PR branch by name (works for same-repo branches)
        if head_ref:
            hr = _safe_ref_name(head_ref)
            try:
                await self._run_git(
                    ["git", "fetch", "origin", f"+refs/heads/{hr}:refs/heads/pr_head/{hr}"],
                    cwd=str(bare_repo),
                )
            except Exception as e:
                logger.warning("Fetch head_ref %s failed: %s", hr, e)

        # Verify SHA exists locally before attempting worktree add
        await self._run_git(["git", "cat-file", "-e", f"{head_sha}^{{commit}}"], cwd=str(bare_repo))

        # Create fresh worktree
        workspace_path = self.base_dir / job_id
        if workspace_path.exists():
            shutil.rmtree(workspace_path, ignore_errors=True)

        logger.info("Creating worktree at %s", workspace_path)
        await self._run_git(
            ["git", "worktree", "add", "--detach", "--force", str(workspace_path), head_sha],
            cwd=str(bare_repo),
        )

        (workspace_path / ACTIVE_SENTINEL).touch()
        return str(workspace_path)

    async def cleanup_workspace(self, job_id: str, bare_repo_name: Optional[str] = None) -> None:
        workspace_path = self.base_dir / job_id
        try:
            sentinel = workspace_path / ACTIVE_SENTINEL
            if sentinel.exists():
                sentinel.unlink()
        except Exception:
            pass

        if bare_repo_name:
            bare_repo = self.cache_dir / f"{bare_repo_name}.git"
            if bare_repo.exists():
                try:
                    await self._run_git(["git", "worktree", "remove", str(workspace_path), "--force"], cwd=str(bare_repo))
                except Exception:
                    pass

        if workspace_path.exists():
            shutil.rmtree(workspace_path, ignore_errors=True)

    async def cleanup_old_workspaces(self, sentinel_ttl_hours: int = 2) -> int:
        ttl_hours = getattr(self.settings, "workspace_ttl_hours", 0) or 0
        if ttl_hours <= 0 or not self.base_dir.exists():
            return 0

        cutoff = time.time() - (ttl_hours * 3600)
        sentinel_cutoff = time.time() - (sentinel_ttl_hours * 3600)
        cleaned = 0

        for path in self.base_dir.iterdir():
            if path.name.startswith("_") or not path.is_dir():
                continue
            try:
                sentinel = path / ACTIVE_SENTINEL
                # Preserve active workspaces with fresh sentinel
                if sentinel.exists() and sentinel.stat().st_mtime > sentinel_cutoff:
                    continue

                if path.stat().st_mtime < cutoff:
                    shutil.rmtree(path, ignore_errors=True)
                    cleaned += 1
            except Exception as e:
                logger.warning("Failed to clean workspace %s: %s", path.name, e)

        return cleaned

    def mark_workspace_inactive(self, workspace_path: str) -> None:
        """Remove the active sentinel so cleanup can proceed."""
        try:
            sentinel = Path(workspace_path) / ACTIVE_SENTINEL
            if sentinel.exists():
                sentinel.unlink()
        except Exception:
            pass

    async def get_diff_stats(self, workspace_path: str) -> DiffStats:
        try:
            out = await self._run_git(["git", "diff", "--numstat"], cwd=workspace_path)
            files_changed = 0
            lines_added = 0
            lines_removed = 0
            for line in (out or "").splitlines():
                parts = line.split("\t")
                if len(parts) >= 3:
                    try:
                        a = 0 if parts[0] == "-" else int(parts[0])
                        r = 0 if parts[1] == "-" else int(parts[1])
                        lines_added += a
                        lines_removed += r
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

    async def commit_and_push(self, workspace_path: str, message: str, branch: str) -> Optional[str]:
        await self._run_git(["git", "add", "-A"], cwd=workspace_path)
        status = await self._run_git(["git", "status", "--porcelain"], cwd=workspace_path)
        if not status.strip():
            return None
        await self._run_git(["git", "commit", "-m", message], cwd=workspace_path)
        commit_sha = (await self._run_git(["git", "rev-parse", "HEAD"], cwd=workspace_path)).strip()
        await self._run_git(["git", "push", "origin", f"HEAD:{branch}"], cwd=workspace_path)
        return commit_sha

    async def _run_git(self, cmd: list[str], cwd: str) -> str:
        env = os.environ.copy()
        env["GIT_TERMINAL_PROMPT"] = "0"
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(
                f"Git command failed: {' '.join(cmd)}\n"
                f"stderr: {stderr.decode(errors='replace')}"
            )
        return stdout.decode(errors="replace")
