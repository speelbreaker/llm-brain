#!/usr/bin/env python3
"""Print a compact, high-signal summary of the latest fidelity run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def main() -> None:
    latest_path = Path("data/fidelity_runs/latest.json")
    if not latest_path.exists():
        raise SystemExit("Missing data/fidelity_runs/latest.json")

    latest = _load_json(latest_path)
    run_id = latest.get("run_id") or latest.get("latest_run_id") or latest.get("id")
    if not run_id:
        raise SystemExit("latest.json missing run id")

    report_path = Path(f"data/fidelity_runs/{run_id}/fidelity_report.json")
    if not report_path.exists():
        raise SystemExit(f"Missing report: {report_path}")

    rep = _load_json(report_path)

    def p(label: str, value: Any) -> None:
        print(f"{label}: {value}")

    print("TOP-LEVEL")
    p("run_id", rep.get("run_id"))
    p("timestamp", rep.get("timestamp"))
    p("underlying", rep.get("underlying"))
    p("overall_score", rep.get("overall_score"))
    p("gate_label", rep.get("gate_label"))
    p("gate_reason", rep.get("gate_reason"))
    coverage = rep.get("coverage") or {}
    p("coverage.coverage_ratio", coverage.get("coverage_ratio"))
    p("coverage.penalty_ratio", coverage.get("penalty_ratio"))
    p("coverage.total_trades_opened", coverage.get("total_trades_opened"))
    p("coverage.valid_trades_closed", coverage.get("valid_trades_closed"))
    p("coverage.invalid_trades_missing_quote", coverage.get("invalid_trades_missing_quote"))

    print("\nREPLAY")
    rd = rep.get("replay_diagnostics") or {}
    for side in ("live", "synthetic"):
        s = rd.get(side) or {}
        if not s:
            continue
        print(f"[{side}]")
        for k in (
            "snapshots_count",
            "spot_min",
            "spot_max",
            "spot_avg",
            "options_count_min",
            "options_count_max",
            "options_count_avg",
        ):
            if k in s:
                p(k, s.get(k))

        fs = s.get("first_snapshot") or {}
        if fs:
            p("first_snapshot.spot", fs.get("spot"))
            p("first_snapshot.options_count", fs.get("options_count"))
            sample = fs.get("sample_options") or []
            if sample:
                p("first_snapshot.sample_option_fields", sorted(sample[0].keys()))

    print("\nSTRATEGIES")
    sd = rep.get("strategy_diagnostics") or {}
    opened_live = {
        name: int(((diag or {}).get("live") or {}).get("opened_trades") or 0)
        for name, diag in sd.items()
    }
    top_opened = sorted(opened_live.items(), key=lambda kv: (-kv[1], kv[0]))[:12]
    p("top_opened_live", top_opened)

    for name, diag in sd.items():
        live = (diag or {}).get("live") or {}
        opened = int(live.get("opened_trades") or 0)
        skips = live.get("skip_reasons") or {}
        if opened == 0 and skips:
            top_skips = sorted(skips.items(), key=lambda kv: (-kv[1], kv[0]))[:6]
            print(f"- {name} top_skips: {top_skips}")


if __name__ == "__main__":
    main()
