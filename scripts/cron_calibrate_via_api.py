#!/usr/bin/env python3
"""Cron-friendly daily calibration via the running web API.

This is the minimal "call API endpoint daily" approach:
- You run this from cron (or any scheduler) on the same host/network as the web app.
- It POSTs to the running FastAPI service endpoint:
    - /api/calibration/run_with_policy  (default)
    - /api/calibration/force_apply      (when --force)

Why this exists:
- Local CLI scripts like scripts/auto_calibrate_daily.py run in a separate process.
  Any in-memory runtime state they set is lost on exit.
- Calling the running service ensures updates happen in the same process that
  serves backtests/pricing.

Default behavior:
- Treats "not applied" as success (policy may choose not to apply).
- Exits non-zero only on HTTP/network errors or invalid responses.

Examples:
  CALIBRATION_BASE_URL="http://localhost:8000" \
    python scripts/cron_calibrate_via_api.py

  CALIBRATION_BASE_URL="https://my-host" CALIBRATION_SOURCE="harvested" \
    CALIBRATION_UNDERLYINGS="BTC,ETH" CALIBRATION_MIN_DTE=3 CALIBRATION_MAX_DTE=10 \
    python scripts/cron_calibrate_via_api.py

Cron example (03:10 UTC):
  10 3 * * * CALIBRATION_BASE_URL="http://127.0.0.1:8000" \
    /path/to/repo/scripts/cron_calibrate_via_api.py >> /path/to/repo/logs/cron_calibrate_api.log 2>&1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ApiResult:
    underlying: str
    ok: bool
    http_status: Optional[int] = None
    applied: Optional[bool] = None
    applied_reason: str = ""
    message: str = ""
    raw: Optional[Dict[str, Any]] = None


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value is not None and value.strip() else default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return float(value)


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return int(value)


def _post_json(url: str, payload: Dict[str, Any], timeout_s: int) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")

    token = os.getenv("CALIBRATION_BEARER_TOKEN")
    if token:
        req.add_header("Authorization", f"Bearer {token}")

    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        body = resp.read().decode("utf-8")
        parsed = json.loads(body) if body else {}
        if not isinstance(parsed, dict):
            raise ValueError("Non-object JSON response")
        parsed["_http_status"] = getattr(resp, "status", None)
        return parsed


def run_one(
    base_url: str,
    endpoint: str,
    underlying: str,
    source: str,
    min_dte: float,
    max_dte: float,
    timeout_s: int,
) -> ApiResult:
    url = base_url.rstrip("/") + endpoint
    payload = {
        "underlying": underlying,
        "source": source,
        "min_dte": float(min_dte),
        "max_dte": float(max_dte),
    }

    try:
        resp = _post_json(url, payload, timeout_s=timeout_s)
        http_status = resp.get("_http_status")

        if resp.get("status") != "ok":
            return ApiResult(
                underlying=underlying,
                ok=False,
                http_status=http_status,
                message=f"API returned non-ok status: {resp.get('status')}",
                raw=resp,
            )

        applied = resp.get("applied")
        applied_reason = resp.get("applied_reason") or ""

        return ApiResult(
            underlying=underlying,
            ok=True,
            http_status=http_status,
            applied=applied if isinstance(applied, bool) else None,
            applied_reason=str(applied_reason),
            raw=resp,
        )

    except urllib.error.HTTPError as e:
        try:
            body = e.read().decode("utf-8")
        except Exception:
            body = ""
        return ApiResult(
            underlying=underlying,
            ok=False,
            http_status=getattr(e, "code", None),
            message=f"HTTPError: {e} {body}".strip(),
        )
    except Exception as e:
        return ApiResult(
            underlying=underlying,
            ok=False,
            message=f"Error: {e}",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Daily calibration via running web API")
    parser.add_argument(
        "--base-url",
        default=_env_str("CALIBRATION_BASE_URL", "http://127.0.0.1:8000"),
        help="Base URL for the running web app (env: CALIBRATION_BASE_URL)",
    )
    parser.add_argument(
        "--underlyings",
        default=_env_str("CALIBRATION_UNDERLYINGS", "BTC,ETH"),
        help="Comma-separated underlyings (env: CALIBRATION_UNDERLYINGS)",
    )
    parser.add_argument(
        "--source",
        default=_env_str("CALIBRATION_SOURCE", "harvested"),
        choices=["live", "harvested"],
        help="Calibration source (env: CALIBRATION_SOURCE)",
    )
    parser.add_argument(
        "--min-dte",
        type=float,
        default=_env_float("CALIBRATION_MIN_DTE", 3.0),
        help="Min DTE (env: CALIBRATION_MIN_DTE)",
    )
    parser.add_argument(
        "--max-dte",
        type=float,
        default=_env_float("CALIBRATION_MAX_DTE", 10.0),
        help="Max DTE (env: CALIBRATION_MAX_DTE)",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=_env_int("CALIBRATION_TIMEOUT_SECONDS", 60),
        help="HTTP timeout seconds (env: CALIBRATION_TIMEOUT_SECONDS)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Call /api/calibration/force_apply (otherwise run_with_policy)",
    )

    args = parser.parse_args()

    endpoint = "/api/calibration/force_apply" if args.force else "/api/calibration/run_with_policy"

    underlyings: List[str] = [u.strip().upper() for u in args.underlyings.split(",") if u.strip()]
    if not underlyings:
        print("ERROR: No underlyings specified")
        sys.exit(2)

    print("============================================================")
    print("Daily Calibration via API")
    print("============================================================")
    print(f"Base URL: {args.base_url}")
    print(f"Endpoint: {endpoint}")
    print(f"Source: {args.source}")
    print(f"DTE band: {args.min_dte}-{args.max_dte}")
    print(f"Underlyings: {','.join(underlyings)}")
    print("============================================================")

    results: List[ApiResult] = []
    for underlying in underlyings:
        r = run_one(
            base_url=args.base_url,
            endpoint=endpoint,
            underlying=underlying,
            source=args.source,
            min_dte=args.min_dte,
            max_dte=args.max_dte,
            timeout_s=args.timeout_seconds,
        )
        results.append(r)

        if r.ok:
            applied_str = "applied" if r.applied else "not applied"
            reason_str = f" ({r.applied_reason})" if r.applied_reason else ""
            print(f"{underlying}: OK, {applied_str}{reason_str}")
        else:
            code_str = f"HTTP {r.http_status}" if r.http_status else ""
            print(f"{underlying}: FAIL {code_str} {r.message}".strip())

    failed = [r for r in results if not r.ok]
    if failed:
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
