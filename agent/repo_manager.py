"""Repo manager for the Telegram Code Review Agent.

Option-B support: review repos that are not cloned locally by cloning/fetching into a
writable cache directory (typically under artifacts/).

Security stance: allow only https://github.com/<owner>/<repo> URLs by default.
"""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple


GITHUB_RE = re.compile(r"^https://github\.com/(?P<owner>[A-Za-z0-9_.-]+)/(?P<repo>[A-Za-z0-9_.-]+?)(?:\.git)?/?$")


@dataclass
class RepoSpec:
    owner: str
    repo: str

    @property
    def slug(self) -> str:
        return f"{self.owner}/{self.repo}"


def parse_github_repo_url(url: str) -> RepoSpec:
    m = GITHUB_RE.match((url or "").strip())
    if not m:
        raise ValueError("Only https://github.com/<owner>/<repo> URLs are supported")
    return RepoSpec(owner=m.group("owner"), repo=m.group("repo"))


def _repo_cache_root() -> Path:
    # Must be writable under systemd ReadWritePaths.
    base = os.environ.get("REPO_CACHE_DIR") or os.environ.get("AUDITOR_DIR")
    if base:
        return Path(base).expanduser().resolve() / "repos"
    return Path("artifacts/.auditor").resolve() / "repos"


def _auth_remote_url(spec: RepoSpec) -> str:
    token = (os.environ.get("GITHUB_TOKEN") or "").strip()
    if token:
        # Do not log this URL anywhere.
        return f"https://x-access-token:{token}@github.com/{spec.owner}/{spec.repo}.git"
    return f"https://github.com/{spec.owner}/{spec.repo}.git"


def ensure_repo_checked_out(url: str, *, depth: int = 200) -> Tuple[Path, RepoSpec]:
    """Clone or update a GitHub repo into the local cache.

    Returns (local_path, RepoSpec).
    """
    spec = parse_github_repo_url(url)
    root = _repo_cache_root()
    root.mkdir(parents=True, exist_ok=True)

    local = root / f"{spec.owner}__{spec.repo}"
    remote = _auth_remote_url(spec)

    if not (local / ".git").exists():
        # Shallow clone for speed; deepen later if needed.
        subprocess.run(
            ["git", "clone", "--depth", str(depth), "--no-tags", remote, str(local)],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
    else:
        # Update existing clone.
        subprocess.run(
            ["git", "fetch", "--prune", "--no-tags", "--depth", str(depth), "origin"],
            check=True,
            cwd=str(local),
            capture_output=True,
            text=True,
            timeout=120,
        )

    # Ensure we have a readable default branch checked out.
    # Prefer origin/HEAD if available.
    try:
        subprocess.run(
            ["git", "checkout", "-q", "--detach"],
            check=True,
            cwd=str(local),
            capture_output=True,
            text=True,
            timeout=30,
        )
    except Exception:
        pass

    return local, spec


def resolve_ref(repo_path: Path, ref: str) -> str:
    """Resolve a ref-ish to a commit sha (raises if invalid)."""
    r = subprocess.run(
        ["git", "rev-parse", ref],
        cwd=str(repo_path),
        capture_output=True,
        text=True,
        timeout=20,
    )
    if r.returncode != 0:
        raise ValueError(f"Invalid ref: {ref}")
    return r.stdout.strip()


def default_compare_range(repo_path: Path) -> Tuple[str, str]:
    """Best-effort default compare range.

    Uses origin/HEAD..HEAD if possible, else HEAD~1..HEAD.
    """
    # Ensure we are on latest origin/HEAD
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(repo_path), capture_output=True, text=True, timeout=10)
        if head.returncode != 0:
            raise RuntimeError
    except Exception:
        return "HEAD~1", "HEAD"

    # Prefer main/master if present.
    for base in ("origin/main", "origin/master", "HEAD~1"):
        try:
            resolve_ref(repo_path, base)
            return base, "HEAD"
        except Exception:
            continue

    return "HEAD~1", "HEAD"
