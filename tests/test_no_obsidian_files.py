import subprocess


def test_no_obsidian_files_tracked():
    """Fail CI if any docs/obsidian entries exist in the public app repo."""
    result = subprocess.run(
        ["git", "ls-files", "docs/obsidian"],
        check=True,
        capture_output=True,
        text=True,
    )
    tracked = [line for line in result.stdout.splitlines() if line.strip()]
    assert not tracked, (
        "docs/obsidian/ must live in the private vault repo. "
        "Remove or migrate these files before committing."
    )
