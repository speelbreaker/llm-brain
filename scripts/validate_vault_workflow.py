#!/usr/bin/env python3

"""Validator for the obsidian vault workflow."""

from __future__ import annotations

import argparse
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Iterable, Set

PROMPT_PATTERN = re.compile(r"(docs/obsidian/06_PROMPTS/[^)\s]+\.md)")


def _repo_root_from_queue(queue_path: Path) -> Path:
    for parent in queue_path.parents:
        if parent.name == "docs":
            return parent.parent
    raise RuntimeError(f"Queue file {queue_path} is not under a docs directory")


def read_queue_prompt_paths(queue_path: Path, repo_root: Path | None = None) -> Set[Path]:
    text = queue_path.read_text()
    matches = PROMPT_PATTERN.findall(text)
    if not matches:
        raise RuntimeError(f"No prompt references found in {queue_path}")
    root = repo_root or _repo_root_from_queue(queue_path)
    return {(root / Path(match)).resolve() for match in matches}


def ensure_prompts_exist(prompt_paths: Iterable[Path]) -> None:
    missing = [path for path in prompt_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Queue references missing prompt files: " + ", ".join(str(p) for p in missing)
        )


def newest_prompt(prompt_dir: Path) -> Path:
    prompts = list(prompt_dir.glob("*.md"))
    if not prompts:
        raise RuntimeError(f"No prompt files under {prompt_dir}")
    return max(prompts, key=lambda path: path.stat().st_mtime)


def ensure_newest_prompt_referenced(
    prompt_paths: Set[Path], prompt_dir: Path
) -> None:
    newest = newest_prompt(prompt_dir).resolve()
    if newest not in prompt_paths:
        raise RuntimeError(
            f"Newest prompt {newest} is not referenced in the queue"
        )


def ensure_changelog_recent(changelog_path: Path, max_age_days: int) -> None:
    for line in changelog_path.read_text().splitlines():
        if line.strip().startswith("- Date:"):
            _, date_str = line.split(":", 1)
            date_str = date_str.strip()
            try:
                entry_date = datetime.fromisoformat(date_str)
            except ValueError:
                entry_date = datetime.fromisoformat(date_str + "T00:00:00")
            now = datetime.now(timezone.utc)
            entry_dt = entry_date.replace(tzinfo=timezone.utc)
            if now - entry_dt > timedelta(days=max_age_days):
                raise RuntimeError(
                    f"Top changelog entry {date_str} is older than {max_age_days} days"
                )
            return
    raise RuntimeError("No dated changelog entry found")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate docs/obsidian queue, prompts, and changelog."
    )
    parser.add_argument(
        "--queue",
        default=Path("docs/obsidian/02_QUEUE/QUEUE.md"),
        type=Path,
        help="Queue file location",
    )
    parser.add_argument(
        "--prompts-dir",
        default=Path("docs/obsidian/06_PROMPTS"),
        type=Path,
        help="Directory containing prompt files",
    )
    parser.add_argument(
        "--changelog",
        default=Path("docs/obsidian/03_LOGS/CHANGELOG.md"),
        type=Path,
        help="Changelog path",
    )
    parser.add_argument(
        "--max-changelog-age-days",
        default=7,
        type=int,
        help="Maximum age of the latest changelog entry in days",
    )
    args = parser.parse_args()

    queue_paths = read_queue_prompt_paths(args.queue)
    ensure_prompts_exist(queue_paths)
    ensure_newest_prompt_referenced(queue_paths, args.prompts_dir)
    ensure_changelog_recent(args.changelog, args.max_changelog_age_days)

    print("Vault workflow validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
