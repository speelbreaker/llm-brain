import json
import os
import subprocess
import sys
from pathlib import Path


def test_gen_ops_health_latest_creates_valid_json_with_required_keys(tmp_path: Path) -> None:
    # Minimal repo guard for generator
    (tmp_path / "pyproject.toml").write_text("[tool]\n", encoding="utf-8")

    script = Path(__file__).resolve().parents[1] / "scripts" / "gen_ops_health_latest.py"
    out_path = tmp_path / "docs" / "OPS_HEALTH_latest.json"

    env = os.environ.copy()
    env["CONTEXT_PACK_FAKE_OPS_HEALTH"] = "1"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--repo-root",
            str(tmp_path),
        ],
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert out_path.exists(), f"Expected {out_path} to be created"

    payload = json.loads(out_path.read_text(encoding="utf-8"))

    # Required keys per spec
    for key in [
        "checked_at",
        "overall_status",
        "worst_severity",
        "can_trade",
        "summary",
        "gates",
        "gate_overall",
        "checks",
    ]:
        assert key in payload

    assert payload["summary"] == "FAKE_OPS_HEALTH"


def test_gen_ops_health_latest_missing_sources_still_succeeds(tmp_path: Path) -> None:
    # Generator should not require any external services in fake mode.
    (tmp_path / "pyproject.toml").write_text("[tool]\n", encoding="utf-8")

    script = Path(__file__).resolve().parents[1] / "scripts" / "gen_ops_health_latest.py"

    env = os.environ.copy()
    env["CONTEXT_PACK_FAKE_OPS_HEALTH"] = "1"

    result = subprocess.run(
        [sys.executable, str(script), "--repo-root", str(tmp_path)],
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )

    assert result.returncode == 0
    assert "Wrote" in result.stdout
