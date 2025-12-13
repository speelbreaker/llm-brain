"""
Change detection module for the Telegram Code Review Agent.

Primary: Git-based detection using git log/diff.
Fallback: File snapshot comparison when git is unavailable.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from agent.storage import get_last_snapshot, get_meta, save_snapshot, set_meta


@dataclass
class ChangedFile:
    """Represents a changed file."""
    path: str
    status: str
    additions: int = 0
    deletions: int = 0
    
    @property
    def status_label(self) -> str:
        """Human-readable status label."""
        labels = {
            "A": "added",
            "M": "modified",
            "D": "deleted",
            "R": "renamed",
            "C": "copied",
        }
        return labels.get(self.status, self.status)


@dataclass
class ChangeResult:
    """Result of change detection."""
    mode: str
    changed_files: List[ChangedFile] = field(default_factory=list)
    diff_text: str = ""
    from_ref: Optional[str] = None
    to_ref: Optional[str] = None
    has_changes: bool = False
    error: Optional[str] = None


class ChangeDetector:
    """Detects changes in the repository."""
    
    EXCLUDE_PATTERNS = [
        ".git",
        ".venv",
        "__pycache__",
        "node_modules",
        ".pyc",
        ".pyo",
        "*.egg-info",
        ".pytest_cache",
        ".mypy_cache",
        "data/",
        "logs/",
        "*.db",
        "*.sqlite",
        "attached_assets/",
    ]
    
    def __init__(self, repo_path: Optional[Path] = None):
        self.repo_path = repo_path or Path(".")
        self._git_available: Optional[bool] = None
    
    def is_git_available(self) -> bool:
        """Check if git is available and the directory is a git repo."""
        if self._git_available is not None:
            return self._git_available
        
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--is-inside-work-tree"],
                capture_output=True,
                text=True,
                cwd=self.repo_path,
                timeout=5,
            )
            self._git_available = result.returncode == 0
        except (subprocess.SubprocessError, FileNotFoundError):
            self._git_available = False
        
        return self._git_available
    
    def get_current_head(self) -> Optional[str]:
        """Get the current HEAD commit hash."""
        if not self.is_git_available():
            return None
        
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                cwd=self.repo_path,
                timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip()[:7]
        except subprocess.SubprocessError:
            pass
        
        return None
    
    def get_recent_changes_since_last_review(self) -> ChangeResult:
        """Get changes since the last reviewed commit."""
        if self.is_git_available():
            return self._get_git_changes()
        else:
            return self._get_snapshot_changes()
    
    def _get_git_changes(self) -> ChangeResult:
        """Get changes using git."""
        last_reviewed = get_meta("last_reviewed_commit")
        current_head = self.get_current_head()
        
        if not current_head:
            return ChangeResult(
                mode="git",
                error="Could not determine current HEAD"
            )
        
        if last_reviewed:
            from_ref = last_reviewed
            to_ref = current_head
            diff_range = f"{last_reviewed}..HEAD"
        else:
            from_ref = None
            to_ref = current_head
            diff_range = "HEAD~1..HEAD"
        
        try:
            name_status = subprocess.run(
                ["git", "diff", "--name-status", diff_range],
                capture_output=True,
                text=True,
                cwd=self.repo_path,
                timeout=30,
            )
            
            if name_status.returncode != 0:
                name_status = subprocess.run(
                    ["git", "show", "--name-status", "--pretty=format:", "HEAD"],
                    capture_output=True,
                    text=True,
                    cwd=self.repo_path,
                    timeout=30,
                )
            
            changed_files = []
            for line in name_status.stdout.strip().split("\n"):
                if not line.strip():
                    continue
                parts = line.split("\t")
                if len(parts) >= 2:
                    status = parts[0][0]
                    path = parts[-1]
                    changed_files.append(ChangedFile(path=path, status=status))
            
            numstat = subprocess.run(
                ["git", "diff", "--numstat", diff_range],
                capture_output=True,
                text=True,
                cwd=self.repo_path,
                timeout=30,
            )
            
            if numstat.returncode == 0:
                stats_map = {}
                for line in numstat.stdout.strip().split("\n"):
                    if not line.strip():
                        continue
                    parts = line.split("\t")
                    if len(parts) >= 3:
                        adds = int(parts[0]) if parts[0].isdigit() else 0
                        dels = int(parts[1]) if parts[1].isdigit() else 0
                        path = parts[2]
                        stats_map[path] = (adds, dels)
                
                for cf in changed_files:
                    if cf.path in stats_map:
                        cf.additions, cf.deletions = stats_map[cf.path]
            
            diff_result = subprocess.run(
                ["git", "diff", "--unified=3", diff_range],
                capture_output=True,
                text=True,
                cwd=self.repo_path,
                timeout=60,
            )
            
            diff_text = diff_result.stdout if diff_result.returncode == 0 else ""
            
            return ChangeResult(
                mode="git",
                changed_files=changed_files,
                diff_text=diff_text,
                from_ref=from_ref,
                to_ref=to_ref,
                has_changes=len(changed_files) > 0,
            )
            
        except subprocess.SubprocessError as e:
            return ChangeResult(
                mode="git",
                error=f"Git command failed: {e}"
            )
    
    def _get_snapshot_changes(self) -> ChangeResult:
        """Get changes by comparing file snapshots."""
        current_snapshot = self._compute_snapshot()
        last_snapshot = get_last_snapshot("last_review")
        
        if not last_snapshot:
            return ChangeResult(
                mode="snapshot",
                has_changes=True,
                error="No previous snapshot. First review will establish baseline."
            )
        
        changed_files = []
        current_paths = set(current_snapshot.keys())
        last_paths = set(last_snapshot.keys())
        
        for path in current_paths - last_paths:
            changed_files.append(ChangedFile(path=path, status="A"))
        
        for path in last_paths - current_paths:
            changed_files.append(ChangedFile(path=path, status="D"))
        
        for path in current_paths & last_paths:
            if current_snapshot[path]["hash"] != last_snapshot[path].get("hash"):
                changed_files.append(ChangedFile(path=path, status="M"))
        
        return ChangeResult(
            mode="snapshot",
            changed_files=changed_files,
            has_changes=len(changed_files) > 0,
        )
    
    def _compute_snapshot(self) -> Dict[str, Dict]:
        """Compute a snapshot of all tracked files."""
        snapshot = {}
        
        for path in self._walk_files():
            try:
                stat = path.stat()
                with open(path, "rb") as f:
                    content = f.read()
                    file_hash = hashlib.md5(content).hexdigest()
                
                snapshot[str(path.relative_to(self.repo_path))] = {
                    "size": stat.st_size,
                    "mtime": stat.st_mtime,
                    "hash": file_hash,
                }
            except (OSError, IOError):
                continue
        
        return snapshot
    
    def _walk_files(self) -> List[Path]:
        """Walk all files, excluding patterns."""
        files = []
        
        for root, dirs, filenames in os.walk(self.repo_path):
            dirs[:] = [d for d in dirs if not self._should_exclude(d)]
            
            for filename in filenames:
                if not self._should_exclude(filename):
                    files.append(Path(root) / filename)
        
        return files
    
    def _should_exclude(self, name: str) -> bool:
        """Check if a file/directory should be excluded."""
        for pattern in self.EXCLUDE_PATTERNS:
            if pattern.startswith("*"):
                if name.endswith(pattern[1:]):
                    return True
            elif pattern.endswith("/"):
                if name == pattern[:-1]:
                    return True
            elif name == pattern or name.startswith(pattern):
                return True
        return False
    
    def mark_reviewed(self, commit_hash: Optional[str] = None) -> None:
        """Mark the current state as reviewed."""
        if self.is_git_available():
            if commit_hash is None:
                commit_hash = self.get_current_head()
            if commit_hash:
                set_meta("last_reviewed_commit", commit_hash)
        else:
            snapshot = self._compute_snapshot()
            save_snapshot("last_review", snapshot)
