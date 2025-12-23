from pathlib import Path


def test_gitignore_ignores_obsidian_and_vscode():
    ignore_content = Path(".gitignore").read_text()
    assert ".obsidian/" in ignore_content, ".gitignore must block .obsidian/ artifacts"
    assert ".vscode/" in ignore_content, ".gitignore must block .vscode/ artifacts"
