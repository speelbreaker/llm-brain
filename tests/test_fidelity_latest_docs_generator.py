import subprocess
import sys
from pathlib import Path

import pytest


def _run_generator(repo_root: Path) -> subprocess.CompletedProcess[str]:
    script = Path(__file__).resolve().parents[1] / "scripts" / "gen_fidelity_latest_docs.py"
    return subprocess.run(
        [sys.executable, str(script), "--repo-root", str(repo_root)],
        text=True,
        capture_output=True,
        check=False,
    )


def test_fidelity_generator_copies_btc_sources(tmp_path: Path) -> None:
    # Minimal repo-root guard expectations
    (tmp_path / "pyproject.toml").write_text("[tool]\n", encoding="utf-8")

    src_dir = tmp_path / "data" / "fidelity_runs" / "BTC" / "latest"
    src_dir.mkdir(parents=True)

    (src_dir / "fidelity_report.json").write_text('{"ok": true}\n', encoding="utf-8")
    (src_dir / "fidelity_report.md").write_text("# BTC Fidelity\n", encoding="utf-8")

    result = _run_generator(tmp_path)
    assert result.returncode == 0, result.stderr

    out_json = tmp_path / "docs" / "FIDELITY_BTC_latest.json"
    out_md = tmp_path / "docs" / "FIDELITY_BTC_latest.md"

    assert out_json.exists()
    assert out_md.exists()

    assert out_json.read_text(encoding="utf-8") == '{"ok": true}\n'
    assert out_md.read_text(encoding="utf-8") == "# BTC Fidelity\n"

    assert "Wrote" in result.stdout


def test_fidelity_generator_missing_sources_does_not_fail(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[tool]\n", encoding="utf-8")

    result = _run_generator(tmp_path)
    assert result.returncode == 0

    # Should warn and not create outputs.
    assert "WARN: missing fidelity source" in result.stdout
    assert not (tmp_path / "docs" / "FIDELITY_BTC_latest.json").exists()
    assert not (tmp_path / "docs" / "FIDELITY_BTC_latest.md").exists()
