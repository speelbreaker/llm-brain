"""Extension helpers for ChangeDetector to support explicit diff ranges.

We keep this separate to avoid disrupting the existing per-repo meta behavior.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

from agent.change_detector import ChangeResult, ChangedFile


def get_git_changes_between(repo_path: Path, from_ref: str, to_ref: str) -> ChangeResult:
    diff_range = f"{from_ref}..{to_ref}"

    try:
        name_status = subprocess.run(
            ["git", "diff", "--name-status", diff_range],
            capture_output=True,
            text=True,
            cwd=repo_path,
            timeout=60,
        )

        changed_files: List[ChangedFile] = []
        for line in name_status.stdout.strip().split("\n"):
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                status = parts[0][0]
                path = parts[-1]
                changed_files.append(ChangedFile(path=path, status=status))

        diff_result = subprocess.run(
            ["git", "diff", "--unified=3", diff_range],
            capture_output=True,
            text=True,
            cwd=repo_path,
            timeout=120,
        )

        diff_text = diff_result.stdout if diff_result.returncode == 0 else ""

        return ChangeResult(
            mode="git",
            changed_files=changed_files,
            diff_text=diff_text,
            from_ref=from_ref,
            to_ref=to_ref,
            has_changes=len(changed_files) > 0,
            error=None,
        )

    except subprocess.SubprocessError as e:
        return ChangeResult(mode="git", error=f"Git command failed: {e}")
