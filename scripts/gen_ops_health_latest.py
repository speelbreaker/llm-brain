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

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _validate_repo_root(root: Path) -> None:
    if not (root / "pyproject.toml").exists():
        raise SystemExit(
            f"Refusing to run: repo root does not look valid (missing pyproject.toml): {root}"
        )


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


def _error_payload(repo_root: Path, exc: Exception) -> Dict[str, Any]:
    # Always emit the keys the API would normally include, so downstream consumers
    # (and tests) can rely on schema presence.
    return {
        "generated_at_utc": _now_utc(),
        "head_sha": _head_sha(repo_root),
        "checked_at": None,
        "cache_age_seconds": None,
        "last_run_at": None,
        "overall_status": "FAIL",
        "checks_overall": "FAIL",
        "checks_summary": None,
        "worst_severity": "FATAL",
        "can_trade": False,
        "summary": "OPS_HEALTH_GENERATION_ERROR",
        "error_code": "OPS_HEALTH_GENERATION_ERROR",
        "error_message": str(exc),
        "checks": [],
        "gates": [],
        "gate_overall": None,
        "can_trade_by_underlying": None,
        "agent_paused_due_to_health": None,
        "error": {"type": type(exc).__name__, "message": str(exc)},
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Generate docs/OPS_HEALTH_latest.json offline (no web server)."
    )
    parser.add_argument(
        "--repo-root",
        default=None,
        help="Repo root (defaults to script parent repo). Used by tests.",
    )
    parser.add_argument(
        "--out-path",
        default=None,
        help="Override output path (defaults to <repo_root>/docs/OPS_HEALTH_latest.json).",
    )

    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve() if args.repo_root else _repo_root()
    _validate_repo_root(repo_root)

    docs_dir = repo_root / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out_path).resolve() if args.out_path else (docs_dir / "OPS_HEALTH_latest.json")

    if (os.environ.get("CONTEXT_PACK_FAKE_OPS_HEALTH") or "").strip().lower() in {"1", "true", "yes"}:
        payload = _fake_payload(repo_root)
    else:
        try:
            payload = _real_payload(repo_root)
        except Exception as e:
            payload = _error_payload(repo_root, e)

    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
