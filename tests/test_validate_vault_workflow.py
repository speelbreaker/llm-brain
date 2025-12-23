from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest

from scripts import validate_vault_workflow as validator


def _setup_vault(tmp_path: Path) -> tuple[Path, Path, Path]:
    queue_dir = tmp_path / "docs" / "obsidian" / "02_QUEUE"
    prompt_dir = tmp_path / "docs" / "obsidian" / "06_PROMPTS"
    changelog_dir = tmp_path / "docs" / "obsidian" / "03_LOGS"
    queue_dir.mkdir(parents=True)
    prompt_dir.mkdir(parents=True)
    changelog_dir.mkdir(parents=True)
    queue_path = queue_dir / "QUEUE.md"
    changelog_path = changelog_dir / "CHANGELOG.md"
    return queue_path, prompt_dir, changelog_path


def test_validator_accepts_queue_referencing_latest_prompt(tmp_path: Path) -> None:
    queue_path, prompt_dir, changelog_path = _setup_vault(tmp_path)

    first_prompt = prompt_dir / "prompt-alpha.md"
    second_prompt = prompt_dir / "prompt-beta.md"
    first_prompt.write_text("alpha")
    second_prompt.write_text("beta")
    # Ensure second prompt is newest by touching it last
    second_prompt.touch()

    queue_path.write_text(
        """
# QUEUE
## IN_PROGRESS
1) [WORKFLOW] docs/obsidian/06_PROMPTS/prompt-alpha.md
## DONE
1) docs/obsidian/06_PROMPTS/prompt-beta.md
"""
    )
    changelog_path.write_text(
        "- Date: 2025-12-23\n  - What changed: test entry.\n"
    )

    paths = validator.read_queue_prompt_paths(queue_path, repo_root=tmp_path)
    validator.ensure_prompts_exist(paths)
    validator.ensure_newest_prompt_referenced(paths, prompt_dir)
    validator.ensure_changelog_recent(changelog_path, max_age_days=30)


def test_validator_fails_when_newest_prompt_missing(tmp_path: Path) -> None:
    queue_path, prompt_dir, changelog_path = _setup_vault(tmp_path)

    old_prompt = prompt_dir / "prompt-old.md"
    new_prompt = prompt_dir / "prompt-new.md"
    old_prompt.write_text("old")
    new_prompt.write_text("new")
    new_prompt.touch()

    queue_path.write_text(
        "# QUEUE\n1) docs/obsidian/06_PROMPTS/prompt-old.md\n"
    )
    changelog_path.write_text("- Date: 2025-12-23\n")

    paths = validator.read_queue_prompt_paths(queue_path, repo_root=tmp_path)
    with pytest.raises(RuntimeError):
        validator.ensure_newest_prompt_referenced(paths, prompt_dir)


def test_validator_fails_on_stale_changelog(tmp_path: Path) -> None:
    queue_path, prompt_dir, changelog_path = _setup_vault(tmp_path)

    prompt = prompt_dir / "prompt.md"
    prompt.write_text("prompt")
    queue_path.write_text("# QUEUE\n1) docs/obsidian/06_PROMPTS/prompt.md\n")

    stale_date = (datetime.now(timezone.utc) - timedelta(days=31)).date().isoformat()
    changelog_path.write_text(f"- Date: {stale_date}\n")

    paths = validator.read_queue_prompt_paths(queue_path)
    validator.ensure_prompts_exist(paths)
    with pytest.raises(RuntimeError):
        validator.ensure_changelog_recent(changelog_path, max_age_days=7)
