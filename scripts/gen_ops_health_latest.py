#!/usr/bin/env python3

"""Generate a machine-readable ops health snapshot for context packs.

Writes: docs/OPS_HEALTH_latest.json

Constraints:
- Must not require the web server to be running.
- Must work headless (e.g., systemd timer).
- Tests must be deterministic and require no network calls.

Environment:
- CONTEXT_PACK_FAKE_OPS_HEALTH=1: write a deterministic stub payload.

CONTRACT (always enforced, even on errors):
- overall_status: one of {"OK", "WARN", "FAIL"} - NEVER null
- can_trade: boolean ALWAYS (False on any error)
- worst_severity: non-null ALWAYS ("CRITICAL" on error)
- summary: non-empty string ALWAYS (includes code + short message on error)
- checks: list ALWAYS (empty list OK)
- gates: list ALWAYS (empty list OK)
- gate_overall: dict ALWAYS ({"status": "FAIL"} on error)

On generation errors, also includes:
- error: {"code": "...", "exception": "...", "traceback": "..."}
"""

from __future__ import annotations

import json
import os
import subprocess
import traceback
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
    """Generate a deterministic stub payload that conforms to the contract."""
    return {
        # Envelope fields
        "generated_at_utc": "2000-01-01T00:00Z",
        "head_sha": "0000000000000000000000000000000000000000",
        # Contract-required fields (all non-null)
        "overall_status": "OK",
        "can_trade": True,
        "worst_severity": "OK",
        "summary": "FAKE_OPS_HEALTH: Deterministic stub for testing",
        "checks": [],
        "gates": [],
        "gate_overall": {"status": "OFF", "message": "fake payload"},
        # Additional context fields
        "checked_at": "2000-01-01T00:00:00+00:00",
        "cache_age_seconds": 0.0,
        "last_run_at": "2000-01-01T00:00:00+00:00",
        "checks_overall": "OK",
        "checks_summary": "",
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


def _truncate_traceback(tb_str: str, max_lines: int = 50) -> str:
    """Truncate traceback to max_lines for readable JSON."""
    lines = tb_str.splitlines()
    if len(lines) <= max_lines:
        return tb_str
    # Keep first half and last half with truncation marker
    half = max_lines // 2
    return "\n".join(
        lines[:half] + [f"... ({len(lines) - max_lines} lines truncated) ..."] + lines[-half:]
    )


def _make_error_payload(repo_root: Path, e: Exception) -> Dict[str, Any]:
    """Create a fail-closed payload that conforms to the contract.
    
    Includes full error details while enforcing all required fields.
    """
    exc_type = type(e).__name__
    exc_msg = str(e)
    tb_str = _truncate_traceback(traceback.format_exc(), max_lines=50)
    
    return {
        # Envelope fields
        "generated_at_utc": _now_utc(),
        "head_sha": _head_sha(repo_root),
        # Contract-required fields (fail-closed)
        "overall_status": "FAIL",
        "can_trade": False,
        "worst_severity": "CRITICAL",
        "summary": f"OPS_HEALTH_GENERATION_ERROR: {exc_type}: {exc_msg}",
        "checks": [],
        "gates": [],
        "gate_overall": {
            "status": "FAIL",
            "message": "generation error",
            "code": "OPS_HEALTH_GENERATION_ERROR",
        },
        # Additional context fields
        "checked_at": None,
        "cache_age_seconds": None,
        "last_run_at": None,
        "checks_overall": "FAIL",
        "checks_summary": "",
        "can_trade_by_underlying": None,
        # Error details
        "error": {
            "code": "OPS_HEALTH_GENERATION_ERROR",
            "exception": f"{exc_type}: {exc_msg}",
            "traceback": tb_str,
        },
    }


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
            # Always write a JSON artifact with full error details and fail-closed fields.
            payload = _make_error_payload(repo_root, e)

    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
