import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.security import env_safety_gate


def test_env_safety_gate_flags_unsafe_toggle(tmp_path: Path):
    unsafe = tmp_path / "unsafe.env"
    unsafe.write_text("SUPERVISOR_DEBUG=1\n", encoding="utf-8")

    findings = env_safety_gate.scan_files([unsafe])

    assert (str(unsafe), "SUPERVISOR_DEBUG") in findings


def test_env_safety_gate_flags_real_secret(tmp_path: Path):
    unsafe = tmp_path / "unsafe.env"
    unsafe.write_text("OPENAI_API_KEY=sk-live-xxx\n", encoding="utf-8")

    findings = env_safety_gate.scan_files([unsafe])

    assert (str(unsafe), "OPENAI_API_KEY") in findings


def test_env_safety_gate_allows_placeholders(tmp_path: Path):
    safe = tmp_path / "safe.env"
    safe.write_text("OPENAI_API_KEY=<REDACTED>\nSUPERVISOR_DEBUG=0\n", encoding="utf-8")

    findings = env_safety_gate.scan_files([safe])

    assert findings == []


def test_env_safety_gate_allows_empty_secret_value(tmp_path: Path):
    safe = tmp_path / "safe.env"
    safe.write_text("OPENAI_API_KEY=\n", encoding="utf-8")

    findings = env_safety_gate.scan_files([safe])

    assert findings == []


def test_default_selection_ignores_docs_and_attachments():
    tracked = [
        Path("docs/SUPERVISOR_VPS_SETUP.md"),
        Path("attached_assets/some.txt"),
        Path("docker/.env.supervisor"),
    ]

    selected = env_safety_gate.select_default_paths(tracked)

    assert Path("docker/.env.supervisor") in selected
    assert Path("docs/SUPERVISOR_VPS_SETUP.md") not in selected
    assert Path("attached_assets/some.txt") not in selected


def test_default_selection_skips_examples():
    tracked = [
        Path("docker/.env.supervisor.example"),
        Path("docker/pr-supervisor.env"),
    ]

    selected = env_safety_gate.select_default_paths(tracked)

    assert Path("docker/pr-supervisor.env") in selected
    assert Path("docker/.env.supervisor.example") not in selected
