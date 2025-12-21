from __future__ import annotations

from datetime import date
from typing import Any, Callable, Dict, List, Optional

from src.ops.gates import GateMode, GateResult, GateStatus


HARVEST_OK_AGE_MIN = 60
HARVEST_WARN_AGE_MIN = 180

CALIBRATION_OK_AGE_HOURS = 36
CALIBRATION_WARN_AGE_HOURS = 72


def make_harvest_gate(
    mode: GateMode,
    *,
    facts: Dict[str, Any],
    required: bool,
    underlying: Optional[str] = None,
    required_dir: Optional[str] = None,
    selection_policy: Optional[str] = None,
    selected_key_reason: Optional[str] = None,
    range_start: Optional[date] = None,
    range_end: Optional[date] = None,
) -> GateResult:
    if mode == GateMode.OFF:
        return GateResult(
            name="harvest",
            mode=mode,
            status=GateStatus.PASS,
            code=None,
            message="Harvest gate disabled by policy.",
            details=dict(facts),
            can_trade=True,
            severity="OK",
            scope="underlying" if underlying else "global",
            underlying=(underlying or None),
        )

    available = bool(facts.get("available"))
    age_minutes = facts.get("age_minutes")
    details: Dict[str, Any] = dict(facts)
    if underlying:
        details["underlying"] = underlying
    if required_dir is not None:
        details["required_dir"] = required_dir
    else:
        if selection_policy is not None:
            details["selection_policy"] = selection_policy
        if selected_key_reason is not None:
            details["selected_key_reason"] = selected_key_reason
    if range_start is not None:
        details["range_start"] = range_start.isoformat()
    if range_end is not None:
        details["range_end"] = range_end.isoformat()

    if not available:
        if required:
            return GateResult(
                name="harvest",
                mode=mode,
                status=GateStatus.FAIL,
                code="NO_HARVESTED_FILES",
                message="No harvested files available.",
                details=details,
                can_trade=False,
                severity="FATAL",
                scope="underlying" if underlying else "global",
                underlying=(underlying or None),
            )
        return GateResult(
            name="harvest",
            mode=mode,
            status=GateStatus.WARN,
            code="NO_HARVESTED_FILES",
            message="No harvested files available (not required by policy).",
            details=details,
            can_trade=True,
            severity="DEGRADED",
            scope="underlying" if underlying else "global",
            underlying=(underlying or None),
        )

    # Explicit selection enforcement to prevent unit mismatch drift.
    selected_key = str(facts.get("selected_key") or "")
    if required_dir is not None and selected_key and selected_key != required_dir:
        return GateResult(
            name="harvest",
            mode=mode,
            status=GateStatus.FAIL,
            code="HARVEST_KEY_MISMATCH",
            message=f"Harvest directory mismatch: selected={selected_key} required={required_dir}.",
            details=details,
            can_trade=False,
            severity="FATAL" if mode == GateMode.BLOCK else "DEGRADED",
            scope="underlying" if underlying else "global",
            underlying=(underlying or None),
        )

    # Range coverage (if provided)
    if range_start is not None and range_end is not None:
        file_count = int(facts.get("file_count") or 0)
        if file_count <= 0:
            return GateResult(
                name="harvest",
                mode=mode,
                status=GateStatus.FAIL,
                code="HARVEST_RANGE_EMPTY",
                message="Harvest exists but no files found for requested date range.",
                details=details,
                can_trade=False,
                severity="FATAL" if required else "DEGRADED",
                scope="underlying" if underlying else "global",
                underlying=(underlying or None),
            )

    if age_minutes is None:
        return GateResult(
            name="harvest",
            mode=mode,
            status=GateStatus.WARN,
            code="HARVEST_AGE_UNKNOWN",
            message="Harvest available but age is unknown.",
            details=details,
            can_trade=True,
            severity="DEGRADED",
            scope="underlying" if underlying else "global",
            underlying=(underlying or None),
        )

    age = float(age_minutes)
    if age <= HARVEST_OK_AGE_MIN:
        return GateResult(
            name="harvest",
            mode=mode,
            status=GateStatus.PASS,
            code=None,
            message="Harvest is fresh.",
            details=details,
            can_trade=True,
            severity="OK",
            scope="underlying" if underlying else "global",
            underlying=(underlying or None),
        )
    if age <= HARVEST_WARN_AGE_MIN:
        return GateResult(
            name="harvest",
            mode=mode,
            status=GateStatus.WARN,
            code="HARVEST_STALE",
            message="Harvest is lagging.",
            details=details,
            can_trade=True,
            severity="DEGRADED",
            scope="underlying" if underlying else "global",
            underlying=(underlying or None),
        )

    # Age > 180
    if required:
        return GateResult(
            name="harvest",
            mode=mode,
            status=GateStatus.FAIL,
            code="HARVEST_STALE",
            message="Harvest is stale.",
            details=details,
            can_trade=False,
            severity="FATAL",
            scope="underlying" if underlying else "global",
            underlying=(underlying or None),
        )

    return GateResult(
        name="harvest",
        mode=mode,
        status=GateStatus.WARN,
        code="HARVEST_STALE",
        message="Harvest is stale (not required by policy).",
        details=details,
        can_trade=True,
        severity="DEGRADED",
        scope="underlying" if underlying else "global",
        underlying=(underlying or None),
    )


def make_fidelity_gate(
    mode: GateMode,
    *,
    facts: Dict[str, Any],
    underlying: Optional[str] = None,
) -> GateResult:
    if mode == GateMode.OFF:
        return GateResult(
            name="fidelity",
            mode=mode,
            status=GateStatus.PASS,
            code=None,
            message="Fidelity gate disabled by policy.",
            details=dict(facts),
            can_trade=True,
            severity="OK",
            scope="underlying" if underlying else "global",
            underlying=(underlying or None),
        )

    available = bool(facts.get("available"))
    gate_label = str(facts.get("gate_label") or "").upper()
    details: Dict[str, Any] = dict(facts)
    if underlying:
        details["underlying"] = underlying

    if not available:
        return GateResult(
            name="fidelity",
            mode=mode,
            status=GateStatus.WARN,
            code="FIDELITY_MISSING",
            message="No fidelity runs found.",
            details=details,
            can_trade=True,
            severity="DEGRADED",
            scope="underlying" if underlying else "global",
            underlying=(underlying or None),
        )

    if gate_label == "TRUSTED":
        return GateResult(
            name="fidelity",
            mode=mode,
            status=GateStatus.PASS,
            code=None,
            message="Fidelity gate is TRUSTED.",
            details=details,
            can_trade=True,
            severity="OK",
            scope="underlying" if underlying else "global",
            underlying=(underlying or None),
        )
    if gate_label == "WARNING":
        return GateResult(
            name="fidelity",
            mode=mode,
            status=GateStatus.WARN,
            code="FIDELITY_WARNING",
            message="Fidelity gate is WARNING.",
            details=details,
            can_trade=True,
            severity="DEGRADED",
            scope="underlying" if underlying else "global",
            underlying=(underlying or None),
        )
    if gate_label == "UNTRUSTED":
        return GateResult(
            name="fidelity",
            mode=mode,
            status=GateStatus.FAIL,
            code="FIDELITY_UNTRUSTED",
            message="Synthetic fidelity UNTRUSTED. Run calibration/fidelity suite first.",
            details=details,
            can_trade=False if mode == GateMode.BLOCK else True,
            severity="FATAL" if mode == GateMode.BLOCK else "DEGRADED",
            scope="underlying" if underlying else "global",
            underlying=(underlying or None),
        )

    return GateResult(
        name="fidelity",
        mode=mode,
        status=GateStatus.WARN,
        code="FIDELITY_UNKNOWN",
        message=f"Fidelity gate is {gate_label or 'UNKNOWN'}.",
        details=details,
        can_trade=True,
        severity="DEGRADED",
        scope="underlying" if underlying else "global",
        underlying=(underlying or None),
    )


def make_calibration_gate(
    mode: GateMode,
    *,
    facts: Dict[str, Any],
    underlying: Optional[str] = None,
) -> GateResult:
    if mode == GateMode.OFF:
        return GateResult(
            name="calibration",
            mode=mode,
            status=GateStatus.PASS,
            code=None,
            message="Calibration gate disabled by policy.",
            details=dict(facts),
            can_trade=True,
            severity="OK",
            scope="underlying" if underlying else "global",
            underlying=(underlying or None),
        )

    available = bool(facts.get("available"))
    age_hours = facts.get("age_hours")
    last_status = str(facts.get("last_status") or "unknown").lower()
    details: Dict[str, Any] = dict(facts)
    if underlying:
        details["underlying"] = underlying

    if not available:
        return GateResult(
            name="calibration",
            mode=mode,
            status=GateStatus.FAIL,
            code="CALIBRATION_MISSING",
            message="Calibration is missing.",
            details=details,
            can_trade=False if mode == GateMode.BLOCK else True,
            severity="FATAL" if mode == GateMode.BLOCK else "DEGRADED",
            scope="underlying" if underlying else "global",
            underlying=(underlying or None),
        )

    if last_status == "failed":
        return GateResult(
            name="calibration",
            mode=mode,
            status=GateStatus.FAIL,
            code="CALIBRATION_FAILED",
            message="Latest calibration run failed.",
            details=details,
            can_trade=False if mode == GateMode.BLOCK else True,
            severity="FATAL" if mode == GateMode.BLOCK else "DEGRADED",
            scope="underlying" if underlying else "global",
            underlying=(underlying or None),
        )

    if age_hours is None:
        return GateResult(
            name="calibration",
            mode=mode,
            status=GateStatus.WARN,
            code="CALIBRATION_AGE_UNKNOWN",
            message="Calibration age is unknown.",
            details=details,
            can_trade=True,
            severity="DEGRADED",
            scope="underlying" if underlying else "global",
            underlying=(underlying or None),
        )

    age = float(age_hours)
    if age <= CALIBRATION_OK_AGE_HOURS and last_status in ("applied", "blocked"):
        if last_status == "blocked":
            return GateResult(
                name="calibration",
                mode=mode,
                status=GateStatus.WARN,
                code="CALIBRATION_BLOCKED",
                message="Calibration is fresh but not applied.",
                details=details,
                can_trade=True,
                severity="DEGRADED",
                scope="underlying" if underlying else "global",
                underlying=(underlying or None),
            )
        return GateResult(
            name="calibration",
            mode=mode,
            status=GateStatus.PASS,
            code=None,
            message="Calibration is fresh and applied.",
            details=details,
            can_trade=True,
            severity="OK",
            scope="underlying" if underlying else "global",
            underlying=(underlying or None),
        )

    if age <= CALIBRATION_WARN_AGE_HOURS or last_status == "blocked":
        code = "CALIBRATION_BLOCKED" if last_status == "blocked" else "CALIBRATION_STALE"
        return GateResult(
            name="calibration",
            mode=mode,
            status=GateStatus.WARN,
            code=code,
            message="Calibration is lagging or blocked.",
            details=details,
            can_trade=True,
            severity="DEGRADED",
            scope="underlying" if underlying else "global",
            underlying=(underlying or None),
        )

    return GateResult(
        name="calibration",
        mode=mode,
        status=GateStatus.FAIL,
        code="CALIBRATION_STALE",
        message="Calibration is stale.",
        details=details,
        can_trade=False if mode == GateMode.BLOCK else True,
        severity="FATAL" if mode == GateMode.BLOCK else "DEGRADED",
        scope="underlying" if underlying else "global",
        underlying=(underlying or None),
    )


def build_underlying_gate_fns(
    *,
    underlying: str,
    harvest_mode: GateMode,
    harvest_required: bool,
    harvest_facts: Dict[str, Any],
    require_harvest_dir: Optional[str],
    fidelity_mode: GateMode,
    fidelity_facts: Dict[str, Any],
    calibration_mode: GateMode,
    calibration_facts: Dict[str, Any],
    range_start: Optional[date] = None,
    range_end: Optional[date] = None,
) -> List[Callable[[], GateResult]]:
    u = (underlying or "").upper().strip()
    selection_policy = "prefer_usdc" if bool(harvest_facts.get("prefer_usdc", True)) else "prefer_inverse"
    selected_key_reason = "first_dir_with_parquets"

    return [
        lambda: make_harvest_gate(
            harvest_mode,
            facts=harvest_facts,
            required=harvest_required,
            underlying=u,
            required_dir=require_harvest_dir,
            selection_policy=selection_policy,
            selected_key_reason=selected_key_reason,
            range_start=range_start,
            range_end=range_end,
        ),
        lambda: make_fidelity_gate(fidelity_mode, facts=fidelity_facts, underlying=u),
        lambda: make_calibration_gate(calibration_mode, facts=calibration_facts, underlying=u),
    ]
