#!/usr/bin/env python3

"""Generate a machine-readable ops health snapshot for context packs.

Writes: docs/OPS_HEALTH_latest.json

Constraints:
- Must not require the web server to be running.
- Must work headless (e.g., systemd timer).
- Tests must be deterministic and require no network calls.

Environment:
- CONTEXT_PACK_FAKE_OPS_HEALTH=1: write a deterministic stub payload.

Output root keys:
- generated_at_utc
- head_sha
- (plus health payload keys such as overall_status / gates / gate_overall)
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _head_sha(repo_root: Path) -> str | None:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(repo_root))
        return out.decode("utf-8").strip() or None
    except Exception:
        return None


def _fake_payload(repo_root: Path) -> Dict[str, Any]:
    return {
        "generated_at_utc": "2000-01-01T00:00Z",
        "head_sha": "0000000000000000000000000000000000000000",
        "checked_at": "2000-01-01T00:00:00+00:00",
        "cache_age_seconds": 0.0,
        "last_run_at": "2000-01-01T00:00:00+00:00",
        "overall_status": "OK",
        "checks_overall": "OK",
        "checks_summary": "",
        "worst_severity": "OK",
        "can_trade": True,
        "summary": "FAKE_OPS_HEALTH",
        "checks": [],
        "gates": [],
        "gate_overall": None,
        "can_trade_by_underlying": None,
        "ops_facts": {},
        "_note": "Deterministic stub written because CONTEXT_PACK_FAKE_OPS_HEALTH=1",
    }


def _real_payload(repo_root: Path) -> Dict[str, Any]:
    from src import healthcheck

    # Prefer cached status if available.
    cached = healthcheck.get_cached_health_status()
    if cached is None:
        # Run and cache once. This may touch external services in real usage.
        # Tests should set CONTEXT_PACK_FAKE_OPS_HEALTH=1 to avoid network.
        healthcheck.run_and_cache_healthcheck(None)

    payload = healthcheck.get_health_status_for_api()

    out: Dict[str, Any] = dict(payload)
    out["generated_at_utc"] = _now_utc()
    out["head_sha"] = _head_sha(repo_root)
    return out


def main() -> int:
    repo_root = _repo_root()
    docs_dir = repo_root / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    out_path = docs_dir / "OPS_HEALTH_latest.json"

    if (os.environ.get("CONTEXT_PACK_FAKE_OPS_HEALTH") or "").strip().lower() in {"1", "true", "yes"}:
        payload = _fake_payload(repo_root)
    else:
        try:
            payload = _real_payload(repo_root)
        except Exception as e:
            # Always write a JSON artifact, even if healthcheck fails.
            payload = {
                "generated_at_utc": _now_utc(),
                "head_sha": _head_sha(repo_root),
                "overall_status": "FAIL",
                "summary": "OPS_HEALTH_GENERATION_ERROR",
                "error": {"type": type(e).__name__, "message": str(e)},
            }

    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
