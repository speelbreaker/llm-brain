from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class GateMode(str, Enum):
    OFF = "off"
    WARN = "warn"
    BLOCK = "block"


class GateStatus(str, Enum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


def _severity_from_status(status: GateStatus) -> str:
    if status == GateStatus.PASS:
        return "OK"
    if status == GateStatus.WARN:
        return "DEGRADED"
    return "FATAL"


@dataclass
class GateResult:
    name: str
    mode: GateMode
    status: GateStatus
    code: Optional[str]
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    can_trade: bool = True
    severity: str = "OK"
    scope: str = "global"  # "global" | "underlying"
    underlying: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.severity:
            self.severity = _severity_from_status(self.status)

    def is_blocking(self) -> bool:
        return self.mode == GateMode.BLOCK and self.status == GateStatus.FAIL

    def is_warn_only(self) -> bool:
        # In warn mode, FAIL is allowed but should be surfaced as a warning.
        return self.mode == GateMode.WARN and self.status in (GateStatus.WARN, GateStatus.FAIL)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        mode = payload.get("mode")
        status = payload.get("status")
        if isinstance(mode, Enum):
            payload["mode"] = mode.value
        if isinstance(status, Enum):
            payload["status"] = status.value
        return payload


class GateRunner:
    def _aggregate(self, results: List[GateResult]) -> Dict[str, Any]:
        any_blocking_fail = any(r.is_blocking() for r in results)
        any_warnish = any(
            (r.status == GateStatus.WARN) or (r.mode == GateMode.WARN and r.status == GateStatus.FAIL)
            for r in results
        )

        if any_blocking_fail:
            overall_status = GateStatus.FAIL
            overall_severity = "FATAL"
        elif any_warnish:
            overall_status = GateStatus.WARN
            overall_severity = "DEGRADED"
        else:
            overall_status = GateStatus.PASS
            overall_severity = "OK"

        can_trade = not any(r.can_trade is False for r in results)

        return {
            "status": overall_status.value,
            "severity": overall_severity,
            "can_trade": can_trade,
        }

    def run(self, gates: List[Callable[[], GateResult]]) -> Dict[str, Any]:
        results: List[GateResult] = [g() for g in gates]

        by_underlying: Dict[str, List[GateResult]] = {}
        for r in results:
            if str(getattr(r, "scope", "global") or "global") == "underlying":
                u = (r.underlying or "").upper().strip()
                if u:
                    by_underlying.setdefault(u, []).append(r)

        by_underlying_overall: Dict[str, Any] = {
            u: self._aggregate(rs) for (u, rs) in sorted(by_underlying.items(), key=lambda kv: kv[0])
        }
        global_overall = self._aggregate(results)

        return {
            "gates": [r.to_dict() for r in results],
            # Back-compat for existing callers/UI
            "overall": dict(global_overall),
            "gate_overall": {
                "global": dict(global_overall),
                "by_underlying": by_underlying_overall,
            },
        }
