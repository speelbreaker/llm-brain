#!/usr/bin/env python3
import json
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


EXCLUDE_DIRS = {
    ".git",
    "__pycache__",
    ".venv",
    "node_modules",
    "dist",
    "build",
}
EXCLUDE_FILE_SUFFIXES = {".pyc"}
ENDPOINT_DECORATOR_RE = re.compile(
    r"@\s*(?:router|app)\.(get|post|put|delete|patch|options|head|trace)\(\s*([\"'])([^\"']+)\2"
)
ENDPOINT_ADD_ROUTE_RE = re.compile(
    r"\.add_api_route\(\s*([\"'])([^\"']+)\1"
)

def utc_iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def git(cmd: list[str]) -> str:
    return subprocess.check_output(["git", *cmd], text=True).strip()


def repo_root() -> Path:
    try:
        return Path(git(["rev-parse", "--show-toplevel"]))
    except subprocess.CalledProcessError:
        return Path.cwd()


def list_tree(root: Path) -> tuple[list[dict], bool]:
    entries: list[dict] = []
    data_present = False
    data_mtime = None
    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = Path(dirpath).relative_to(root)
        dirnames[:] = [
            d
            for d in dirnames
            if d not in EXCLUDE_DIRS and not d.startswith(".")
        ]
        if rel_dir == Path(".") and "data" in dirnames:
            data_present = True
            data_dir = root / "data"
            try:
                data_mtime = data_dir.stat().st_mtime
            except OSError:
                data_mtime = None
            dirnames.remove("data")
        if "data" in dirnames:
            dirnames.remove("data")
        for filename in filenames:
            if any(filename.endswith(suffix) for suffix in EXCLUDE_FILE_SUFFIXES):
                continue
            file_path = Path(dirpath) / filename
            rel_path = file_path.relative_to(root)
            try:
                stat = file_path.stat()
            except OSError:
                continue
            entries.append(
                {
                    "path": rel_path.as_posix(),
                    "size_bytes": stat.st_size,
                    "mtime_utc": utc_iso(stat.st_mtime),
                }
            )
    if data_present:
        entries.append(
            {
                "path": "data/",
                "size_bytes": 0,
                "mtime_utc": utc_iso(data_mtime or time.time()),
            }
        )
    return entries, data_present


def find_important_paths(root: Path) -> list[str]:
    candidates = [
        "src",
        "tests",
        "docs",
        "src/ops",
        "src/fidelity",
        "src/backtest",
        "src/web",
    ]
    found: list[str] = []
    for candidate in candidates:
        path = root / candidate
        if path.exists():
            found.append(candidate.rstrip("/") + ("/" if path.is_dir() else ""))
    for pattern in ["ROADMAP_BACKLOG*.md", "HEALTHCHECK*.md"]:
        for match in root.glob(pattern):
            if match.is_file():
                found.append(match.name)
    return sorted(set(found))


def find_endpoints(root: Path, file_entries: list[dict]) -> list[dict]:
    mapping: dict[str, set[str]] = {}
    for entry in file_entries:
        if not entry["path"].endswith(".py"):
            continue
        file_path = root / entry["path"]
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            continue
        for line in content:
            dec_match = ENDPOINT_DECORATOR_RE.search(line)
            if dec_match:
                path = dec_match.group(3)
                mapping.setdefault(path, set()).add(entry["path"])
            add_match = ENDPOINT_ADD_ROUTE_RE.search(line)
            if add_match:
                path = add_match.group(2)
                mapping.setdefault(path, set()).add(entry["path"])
    return [
        {"path": path, "files": sorted(files)}
        for path, files in sorted(mapping.items())
    ]


def main() -> int:
    root = repo_root()
    entries, data_present = list_tree(root)
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    try:
        branch = git(["rev-parse", "--abbrev-ref", "HEAD"])
        head_sha = git(["rev-parse", "HEAD"])
        head_summary = git(["log", "-1", "--pretty=%s"])
        is_dirty = bool(git(["status", "--porcelain"]))
    except subprocess.CalledProcessError:
        branch = "unknown"
        head_sha = "unknown"
        head_summary = ""
        is_dirty = False

    entries_sorted = sorted(entries, key=lambda e: e.get("mtime_utc") or "", reverse=True)
    hotspots = [
        {
            "path": entry["path"],
            "size_bytes": entry["size_bytes"],
            "mtime_utc": entry["mtime_utc"],
        }
        for entry in entries_sorted[:20]
    ]

    manifest = {
        "repo_root": str(root),
        "generated_at_utc": now,
        "git": {
            "branch": branch,
            "head_sha": head_sha,
            "head_summary": head_summary,
            "is_dirty": is_dirty,
        },
        "tree": sorted(entries, key=lambda e: e["path"]),
        "important_paths": find_important_paths(root),
        "hotspots": hotspots,
        "endpoints_index": find_endpoints(root, entries),
        "data_present": data_present,
    }

    output_path = root / "docs" / "REPO_MANIFEST.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
