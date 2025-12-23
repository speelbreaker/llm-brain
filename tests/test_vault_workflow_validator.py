import os
import shutil
import pytest
from pathlib import Path
from scripts.validate_vault_workflow import check_structure, parse_queue, validate_queue, validate_prompts

@pytest.fixture
def mock_vault(tmp_path):
    """Create a mock vault structure in a temp directory."""
    cwd = Path.cwd()
    os.chdir(tmp_path)
    
    vault_root = tmp_path / "docs/obsidian"
    vault_root.mkdir(parents=True)
    
    # Create required dirs
    (vault_root / "00_HOME").mkdir()
    (vault_root / "01_RULES").mkdir()
    (vault_root / "02_QUEUE").mkdir()
    (vault_root / "03_LOGS").mkdir()
    (vault_root / "06_PROMPTS").mkdir()
    (vault_root / "99_ARCHIVE").mkdir()
    
    # Create required files
    (vault_root / "00_HOME/NORTHSTAR.md").touch()
    (vault_root / "01_RULES/QUEUE_DISCIPLINE.md").touch()
    (vault_root / "01_RULES/PR_WORKFLOW.md").touch()
    
    yield vault_root
    
    os.chdir(cwd)

def test_structure_check_passes(mock_vault):
    # Setup minimal valid state
    (mock_vault / "02_QUEUE/QUEUE.md").touch()
    (mock_vault / "06_PROMPTS/_ACTIVE.md").touch()
    
    # Should not raise
    check_structure()

def test_structure_check_fails_missing_dir(mock_vault):
    shutil.rmtree(mock_vault / "03_LOGS")
    with pytest.raises(SystemExit):
        check_structure()

def test_queue_parsing(mock_vault):
    q = mock_vault / "02_QUEUE/QUEUE.md"
    q.write_text("""
## IN_PROGRESS
- item 1

## READY
- item 2
- item 3
""" )
    sections = parse_queue()
    assert len(sections["IN_PROGRESS"]) == 1
    assert len(sections["READY"]) == 2
    assert len(sections["IN_REVIEW"]) == 0

def test_validate_queue_limits(mock_vault, capsys):
    sections = {
        "IN_PROGRESS": ["- 1", "- 2"],
        "READY": [], 
        "IN_REVIEW": [],
        "DONE": []
    }
    with pytest.raises(SystemExit):
        validate_queue(sections)
    
    captured = capsys.readouterr()
    assert "Too many IN_PROGRESS" in captured.out

def test_validate_in_progress_format(mock_vault):
    sections = {
        "IN_PROGRESS": ["- Task | branch: feat/x"], # missing prompt
        "READY": [], "IN_REVIEW": [], "DONE": []
    }
    with pytest.raises(SystemExit):
        validate_queue(sections)

def test_validate_active_pointer(mock_vault):
    sections = {
        "IN_PROGRESS": ["- Task | branch: feat/x | prompt: docs/obsidian/06_PROMPTS/p1.md"],
        "READY": [], "IN_REVIEW": [], "DONE": []
    }
    
    # Create prompt file
    p = mock_vault / "06_PROMPTS/p1.md"
    p.write_text("## Acceptance Criteria\n## Tests / Verification")
    
    # Mismatch _ACTIVE
    (mock_vault / "06_PROMPTS/_ACTIVE.md").write_text("docs/obsidian/06_PROMPTS/other.md")
    
    with pytest.raises(SystemExit):
        validate_queue(sections)
    
    # Match _ACTIVE
    (mock_vault / "06_PROMPTS/_ACTIVE.md").write_text("docs/obsidian/06_PROMPTS/p1.md")
    
    # Should pass (mocking cwd relative check is tricky without full integration, 
    # relying on validate_queue logic which checks Path.cwd() / prompt_path)
    # Since we changed cwd to tmp_path, prompt path "docs/obsidian..." works relative to root.
    validate_queue(sections)

def test_validate_prompts_content(mock_vault):
    sections = {
        "IN_PROGRESS": [],
        "READY": ["- Task | prompt: docs/obsidian/06_PROMPTS/bad.md"],
        "IN_REVIEW": [], "DONE": []
    }
    
    p = mock_vault / "06_PROMPTS/bad.md"
    p.write_text("Just some text")
    
    with pytest.raises(SystemExit):
        validate_prompts(sections)
    
    p.write_text("## Acceptance Criteria\n## Tests / Verification")
    validate_prompts(sections)
