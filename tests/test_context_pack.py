import json
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_gen_repo_manifest_runs_and_valid_json():
    script_path = REPO_ROOT / "scripts" / "gen_repo_manifest.py"
    output_path = REPO_ROOT / "docs" / "REPO_MANIFEST.json"

    subprocess.check_call([sys.executable, str(script_path)], cwd=REPO_ROOT)
    assert output_path.exists()

    data = json.loads(output_path.read_text(encoding="utf-8"))
    assert "repo_root" in data
    assert "generated_at_utc" in data
    assert "git" in data
    assert "tree" in data
    assert "important_paths" in data
    assert "hotspots" in data
    assert "endpoints_index" in data

    git_info = data["git"]
    assert "branch" in git_info
    assert "head_sha" in git_info
    assert "head_summary" in git_info
    assert "is_dirty" in git_info

    if data["tree"]:
        entry = data["tree"][0]
        assert "path" in entry
        assert "size_bytes" in entry
        assert "mtime_utc" in entry


@pytest.mark.skipif(sys.platform.startswith("win"), reason="shell script not supported on Windows")
def test_gen_recent_diff_runs():
    script_path = REPO_ROOT / "scripts" / "gen_recent_diff.sh"
    output_path = REPO_ROOT / "docs" / "RECENT_DIFF.md"

    subprocess.check_call(["bash", str(script_path)], cwd=REPO_ROOT)
    assert output_path.exists()

    content = output_path.read_text(encoding="utf-8")
    assert "generated_at_utc:" in content
    assert "git log --oneline -n 25" in content


def test_gen_ops_health_latest_fake_mode(tmp_path, monkeypatch):
    script_path = REPO_ROOT / "scripts" / "gen_ops_health_latest.py"
    output_path = REPO_ROOT / "docs" / "OPS_HEALTH_latest.json"

    monkeypatch.setenv("CONTEXT_PACK_FAKE_OPS_HEALTH", "1")
    subprocess.check_call([sys.executable, str(script_path)], cwd=REPO_ROOT)
    assert output_path.exists()

    data = json.loads(output_path.read_text(encoding="utf-8"))
    assert data.get("generated_at_utc")
    assert data.get("head_sha")
    # Schema is allowed to evolve; require at least a status key.
    assert ("overall_status" in data) or ("gate_overall" in data)
