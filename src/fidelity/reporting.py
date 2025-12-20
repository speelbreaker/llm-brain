from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class FidelityReport:
    run_id: str
    timestamp: str

    # New (MVP spec)
    underlying: str = "BTC"
    start_ts: int = 0
    end_ts: int = 0
    gate_label: str = "UNTRUSTED"
    live_data_status: str = "missing"  # ok|missing

    # New: richer report payloads
    component_status: Dict[str, str] = field(default_factory=dict)
    components: Dict[str, Any] = field(default_factory=dict)
    per_strategy: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    market_live_meta: Dict[str, Any] = field(default_factory=dict)
    market_synth_meta: Dict[str, Any] = field(default_factory=dict)

    # Keep these for backwards compatibility with the existing UI.
    component_scores: Dict[str, float] = field(default_factory=dict)
    overall_score: float = 0.0
    gate: str = "UNTRUSTED"

    strategy_parity: Dict[str, Any] = field(default_factory=dict)


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
    lines.append(f"- Gate: **{report.gate_label or report.gate}**")
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
    lines.append("## Strategy Parity")
    lines.append("")
    lines.append("```json")
    payload = report.strategy_parity or report.per_strategy or {}
    lines.append(json.dumps(payload, indent=2, sort_keys=True))
    lines.append("```")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_latest_index(report: FidelityReport, path: Path) -> None:
    """Write the global latest.json index expected by the MVP endpoints."""
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "run_id": report.run_id,
        "timestamp": report.timestamp,
        "underlying": report.underlying,
        "start_ts": report.start_ts,
        "end_ts": report.end_ts,
        "overall_score": report.overall_score,
        "gate_label": report.gate_label or report.gate,
        "component_scores": report.component_scores,
        "live_data_status": report.live_data_status,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

