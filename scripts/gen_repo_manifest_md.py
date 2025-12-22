#!/usr/bin/env python3
import json
from pathlib import Path
from datetime import datetime, timezone


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    repo_root = (script_dir / "..").resolve()
    docs_dir = repo_root / "docs"
    manifest_path = docs_dir / "REPO_MANIFEST.json"
    if not manifest_path.exists():
        raise SystemExit(f"Missing {manifest_path}. Run gen_repo_manifest.py first.")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    generated_at = manifest.get("generated_at_utc") or datetime.now(timezone.utc).isoformat()
    git_info = manifest.get("git", {})
    important_paths = manifest.get("important_paths", [])
    endpoints = manifest.get("endpoints_index", [])
    hotspots = manifest.get("hotspots", [])
    tree = manifest.get("tree", [])

    lines = [
        "# REPO_MANIFEST (Latest)",
        "",
        f"- generated_at_utc: {generated_at}",
        f"- repo_root: {manifest.get('repo_root', str(repo_root))}",
        f"- branch: {git_info.get('branch', '')}",
        f"- head_sha: {git_info.get('head_sha', '')}",
        f"- head_summary: {git_info.get('head_summary', '')}",
        f"- is_dirty: {git_info.get('is_dirty', False)}",
        "",
        "## Important paths",
    ]
    for path in important_paths:
        lines.append(f"- {path}")

    lines.extend(["", "## Endpoints index (top)"])
    max_endpoints = 200
    for endpoint in endpoints[:max_endpoints]:
        method = endpoint.get("method", "UNKNOWN")
        path = endpoint.get("path", "")
        file_path = endpoint.get("file", "")
        lines.append(f"- {method} {path} ({file_path})")
    if len(endpoints) > max_endpoints:
        lines.append(f"- TRUNCATED: {len(endpoints) - max_endpoints} more endpoints")

    lines.extend(["", "## Hotspots (recently modified)"])
    for item in hotspots:
        lines.append(
            f"- {item.get('path', '')} ({item.get('size_bytes', 0)} bytes, {item.get('mtime_utc', '')})"
        )

    lines.extend(["", "## Tree summary (top 500)"])
    max_tree = 500
    for item in tree[:max_tree]:
        lines.append(
            f"- {item.get('path', '')} ({item.get('size_bytes', 0)} bytes, {item.get('mtime_utc', '')})"
        )
    if len(tree) > max_tree:
        lines.append(f"- TRUNCATED: {len(tree) - max_tree} more entries")

    out_path = docs_dir / "REPO_MANIFEST_latest.md"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
