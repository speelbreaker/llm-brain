from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest


def test_gen_ops_health_latest(monkeypatch: pytest.MonkeyPatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    docs_path = repo_root / "docs" / "OPS_HEALTH_latest.json"
    env = os.environ.copy()
    env["CONTEXT_PACK_FAKE_OPS_HEALTH"] = "1"

    subprocess.run(
        ["python3", "scripts/gen_ops_health_latest.py"],
        check=True,
        cwd=repo_root,
        env=env,
    )

    assert docs_path.exists()
    data = json.loads(docs_path.read_text(encoding="utf-8"))
    keys = {
        "generated_at_utc",
        "head_sha",
        "overall_status",
        "checks_overall",
        "worst_severity",
        "can_trade",
        "summary",
        "checks",
        "gates",
        "gate_overall",
    }
    assert keys.issubset(data.keys())
    assert data["overall_status"] == "OK"
    assert data["can_trade"] is True
