import pytest
from scripts.supervisor_loop import parse_item, build_item, slugify, parse_queue

def test_parse_item():
    raw = "- P0-001 | My Task | prompt: p.md | branch: b1"
    data = parse_item(raw)
    assert data["id"] == "P0-001"
    assert data["task"] == "My Task"
    assert data["prompt"] == "p.md"
    assert data["branch"] == "b1"

def test_build_item():
    data = {
        "id": "T1", "task": "Task One", "branch": "b1", "prompt": "p1", "agent": "a1",
        "custom": "val"
    }
    s = build_item(data)
    assert s == "- T1 | Task One | branch: b1 | prompt: p1 | agent: a1 | custom: val"

def test_slugify():
    assert slugify("My Task Name!") == "my-task-name"
    assert slugify("Fix #123") == "fix-123"

def test_parse_queue_structure(tmp_path):
    q = tmp_path / "QUEUE.md"
    q.write_text("""
## IN_PROGRESS
- item 1

## READY
- item 2
""")
    # We mock the global QUEUE_FILE in the script logic, but here we test the logic via helper if we could inject content.
    # Since parse_queue reads from global constant in the module, we need to monkeypatch.
    import scripts.supervisor_loop
    scripts.supervisor_loop.QUEUE_FILE = q
    
    sections = scripts.supervisor_loop.parse_queue()
    assert len(sections["IN_PROGRESS"]) == 1
    assert len(sections["READY"]) == 1
    assert sections["READY"][0] == "- item 2"
