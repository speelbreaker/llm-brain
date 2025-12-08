#!/usr/bin/env python3
"""
Agent Healthcheck CLI Script

Run this to verify the agent's critical systems are healthy:
  - Configuration validation
  - Deribit public API connectivity
  - Deribit private API connectivity (if credentials configured)
  - State builder pipeline

Usage:
    uv run python scripts/agent_healthcheck.py

Exit codes:
    0 - All checks passed (OK or WARN)
    1 - At least one check failed (FAIL)
"""
from __future__ import annotations

import sys


def main() -> None:
    from src.config import settings
    from src.healthcheck import run_agent_healthcheck

    print("=" * 60)
    print("Options Trading Agent – Healthcheck")
    print("=" * 60)

    result = run_agent_healthcheck(settings)

    overall = result["overall_status"]
    print(f"\nOverall status: {overall}")
    print("-" * 60)

    for r in result["results"]:
        name = r["name"]
        status = r["status"]
        detail = r["detail"]
        print(f"[{status.upper():7}] {name:18} - {detail}")

    print("-" * 60)

    if overall == "FAIL":
        print("\nHealthcheck FAILED. Please review the errors above.")
        sys.exit(1)
    elif overall == "WARN":
        print("\nHealthcheck passed with WARNINGS. Agent can still run.")
        sys.exit(0)
    else:
        print("\nHealthcheck PASSED. All systems operational.")
        sys.exit(0)


if __name__ == "__main__":
    main()
