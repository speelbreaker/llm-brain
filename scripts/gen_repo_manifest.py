#!/usr/bin/env python3
import json
import os
import re
import subprocess
from pathlib import Path
from datetime import datetime, timezone

RE_EXCLUDE_DIR = re.compile(r"(^\.git$|^__pycache__$|^\.venv$|^venv$|^node_modules$|^dist$|^build$|^data$)")
RE_EXCLUDE_FILE = re.compile(r".*\.pyc$")


def utc_iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def sh(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return ""


def resolve_repo_root(candidate: Path) -> Path:
    try:
        root = subprocess.check_output(
            ["git", "-C", str(candidate), "rev-parse", "--show-toplevel"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        if root:
            return Path(root)
    except Exception:
        pass
    return candidate.resolve()


def detect_endpoints(src_dir: Path) -> list[dict]:
    endpoints = []
    # Very lightweight, best-effort extraction of FastAPI routes.
    deco = re.compile(r"@\s*router\.(get|post|put|delete|patch)\(\s*([\"'])(.+?)\2")
    add = re.compile(r"\.add_api_route\(\s*([\"'])(.+?)\1\s*,")
    for p in src_dir.rglob("*.py"):
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for m in deco.finditer(txt):
            endpoints.append({"method": m.group(1).upper(), "path": m.group(3), "file": str(p)})
        for m in add.finditer(txt):
            endpoints.append({"method": "UNKNOWN", "path": m.group(2), "file": str(p)})
    # Dedup
    seen = set()
    out = []
    for e in endpoints:
        k = (e["method"], e["path"], e["file"])
        if k in seen:
            continue
        seen.add(k)
        out.append(e)
    return out


def main() -> None:
    script_repo_root = Path(__file__).resolve().parents[1]
    repo_root = resolve_repo_root(script_repo_root)
    docs_dir = repo_root / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)

    git = {
        "branch": sh(["git", "-C", str(repo_root), "rev-parse", "--abbrev-ref", "HEAD"]),
        "head_sha": sh(["git", "-C", str(repo_root), "rev-parse", "HEAD"]),
        "head_summary": sh(["git", "-C", str(repo_root), "log", "-1", "--oneline"]),
        "is_dirty": sh(["git", "-C", str(repo_root), "status", "--porcelain"]) != "",
    }

    important_paths = []
    for rel in [
        "src",
        "tests",
        "docs",
        "src/ops",
        "src/fidelity",
        "src/backtest",
        "src/web",
        "ROADMAP_BACKLOG.md",
        "HEALTHCHECK.md",
    ]:
        if (repo_root / rel).exists():
            important_paths.append(rel)

    files = []
    for root, dirs, fnames in os.walk(repo_root):
        dirs[:] = [d for d in dirs if not RE_EXCLUDE_DIR.match(d)]
        for fn in fnames:
            if RE_EXCLUDE_FILE.match(fn):
                continue
            p = Path(root) / fn
            try:
                st = p.stat()
            except Exception:
                continue
            rel = str(p.relative_to(repo_root))
            files.append({
                "path": rel,
                "size_bytes": st.st_size,
                "mtime_utc": utc_iso(st.st_mtime),
            })

    hotspots = sorted(files, key=lambda x: x["mtime_utc"], reverse=True)[:20]

    endpoints = []
    src_dir = repo_root / "src"
    if src_dir.exists():
        endpoints = detect_endpoints(src_dir)

    manifest = {
        "repo_root": str(repo_root),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": git,
        "important_paths": important_paths,
        "hotspots": hotspots,
        "endpoints_index": endpoints,
        "tree": files,
    }

    out_path = docs_dir / "REPO_MANIFEST.json"
    out_path.write_text(json.dumps(manifest, indent=2, sort_keys=False), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
