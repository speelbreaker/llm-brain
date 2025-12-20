import tempfile
from pathlib import Path

from scripts.secret_tripwire import scan_paths


def test_detects_fake_secret():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "config.py"
        path.write_text("OPENAI_API_KEY=sk-THISISFAKEKEYSHOULDALERT\n", encoding="utf-8")
        findings = scan_paths([path])
        assert len(findings) == 1
        assert findings[0]["path"].endswith("config.py")
        assert findings[0]["value"].startswith("***REDACTED***")


def test_ignores_benign_text():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "notes.txt"
        path.write_text("just some harmless text", encoding="utf-8")
        findings = scan_paths([path])
        assert findings == []
