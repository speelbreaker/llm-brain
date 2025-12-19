from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class FidelityReport:
    run_id: str
    timestamp: str

    market_live_meta: Dict[str, Any]
    market_synth_meta: Dict[str, Any]

    component_scores: Dict[str, float]
    overall_score: float
    gate: str

    strategy_parity: Dict[str, Any]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_report_json(report: FidelityReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(report), f, indent=2, sort_keys=True)


def write_report_md(report: FidelityReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append(f"# Synthetic Fidelity Report")
    lines.append("")
    lines.append(f"- Run ID: {report.run_id}")
    lines.append(f"- Timestamp (UTC): {report.timestamp}")
    lines.append(f"- Gate: **{report.gate}**")
    lines.append("")
    lines.append(f"## Scores")
    lines.append("")
    lines.append(f"- Overall: **{report.overall_score:.1f}**")
    for k, v in sorted(report.component_scores.items()):
        lines.append(f"- {k}: {v:.1f}")

    lines.append("")
    lines.append("## Market Meta")
    lines.append("")
    lines.append("### Live")
    lines.append("```json")
    lines.append(json.dumps(report.market_live_meta, indent=2, sort_keys=True))
    lines.append("```")
    lines.append("")
    lines.append("### Synthetic")
    lines.append("```json")
    lines.append(json.dumps(report.market_synth_meta, indent=2, sort_keys=True))
    lines.append("```")

    lines.append("")
    lines.append("## Strategy Parity (P0 placeholder)")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(report.strategy_parity, indent=2, sort_keys=True))
    lines.append("```")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
