import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.security import env_safety_gate


def test_env_safety_gate_flags_unsafe_toggles(tmp_path: Path):
    unsafe = tmp_path / "unsafe.env"
    unsafe.write_text("SUPERVISOR_DEBUG=1\n", encoding="utf-8")

    findings = env_safety_gate.scan_files([unsafe])

    assert (str(unsafe), "SUPERVISOR_DEBUG") in findings


def test_env_safety_gate_flags_secret_assignments(tmp_path: Path):
    unsafe = tmp_path / "unsafe.env"
    unsafe.write_text("OPENAI_API_KEY=dummy\n", encoding="utf-8")

    findings = env_safety_gate.scan_files([unsafe])

    assert (str(unsafe), "OPENAI_API_KEY") in findings


def test_env_safety_gate_allows_safe_files(tmp_path: Path):
    safe = tmp_path / "safe.env"
    safe.write_text("SUPERVISOR_DEBUG 0\n", encoding="utf-8")

    findings = env_safety_gate.scan_files([safe])

    assert findings == []
