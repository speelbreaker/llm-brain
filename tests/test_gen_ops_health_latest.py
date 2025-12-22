from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def test_gen_ops_health_latest_fake_payload_has_required_keys_and_is_deterministic(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "gen_ops_health_latest.py"
    out_path = repo_root / "docs" / "OPS_HEALTH_latest.json"

    prev = out_path.read_text(encoding="utf-8") if out_path.exists() else None

    env = dict(os.environ)
    env["CONTEXT_PACK_FAKE_OPS_HEALTH"] = "1"

    try:
        subprocess.check_call([sys.executable, str(script_path)], env=env, cwd=str(repo_root))
        payload = json.loads(out_path.read_text(encoding="utf-8"))

        # Context-pack envelope
        assert payload.get("generated_at_utc")
        assert payload.get("head_sha")

        # API-shaped health payload (minimum expected keys)
        for key in (
            "checked_at",
            "last_run_at",
            "cache_age_seconds",
            "overall_status",
            "checks_overall",
            "checks_summary",
            "worst_severity",
            "can_trade",
            "summary",
            "checks",
            "gates",
            "gate_overall",
            "can_trade_by_underlying",
            "ops_facts",
        ):
            assert key in payload

        assert payload["summary"] == "FAKE_OPS_HEALTH"
        assert payload["overall_status"] == "OK"
        assert payload["checks_overall"] == "OK"
        assert payload["worst_severity"] == "OK"
        assert payload["can_trade"] is True

        # Fully deterministic in fake mode
        assert payload["generated_at_utc"] == "2000-01-01T00:00Z"
        assert payload["head_sha"] == "0000000000000000000000000000000000000000"
        assert payload["checked_at"] == "2000-01-01T00:00:00+00:00"
    finally:
        if prev is None:
            out_path.unlink(missing_ok=True)
        else:
            out_path.write_text(prev, encoding="utf-8")
