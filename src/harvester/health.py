"""
Harvester data quality and schema validation module.

Provides:
- Schema validation for harvested Parquet snapshots
- Snapshot-level quality assessment
- Aggregated quality summaries for calibration
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


REQUIRED_COLUMNS: Dict[str, str] = {
    "harvest_time": "datetime64",
    "instrument_name": "string",
    "underlying": "string",
    "expiry": "string",
    "expiry_timestamp": "int64",
    "option_type": "string",
    "strike": "float64",
    "underlying_price": "float64",
    "mark_price": "float64",
    "best_bid_price": "float64",
    "best_ask_price": "float64",
    "mark_iv": "float64",
    "open_interest": "float64",
    "volume": "float64",
    "greek_delta": "float64",
    "greek_vega": "float64",
}

CORE_FIELDS = [
    "underlying_price",
    "mark_price",
    "mark_iv",
    "expiry_timestamp",
    "option_type",
]

QUALITY_THRESHOLDS = {
    "min_rows": 1,
    "core_field_fraction": 0.90,
    "degraded_snapshot_fraction": 0.20,
}


class DataQualityStatus(str, Enum):
    OK = "ok"
    DEGRADED = "degraded"
    FAILED = "failed"


@dataclass
class SnapshotQualityReport:
    """Quality report for a single snapshot."""
    
    filename: str = ""
    total_rows: int = 0
    non_null_core_fraction: float = 0.0
    schema_issues: List[str] = field(default_factory=list)
    quality_issues: List[str] = field(default_factory=list)
    
    @property
    def is_schema_valid(self) -> bool:
        return len(self.schema_issues) == 0
    
    @property
    def is_quality_ok(self) -> bool:
        return len(self.quality_issues) == 0
    
    @property
    def status(self) -> DataQualityStatus:
        if not self.is_schema_valid:
            return DataQualityStatus.FAILED
        if not self.is_quality_ok:
            return DataQualityStatus.DEGRADED
        return DataQualityStatus.OK
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "filename": self.filename,
            "total_rows": self.total_rows,
            "non_null_core_fraction": round(self.non_null_core_fraction, 4),
            "schema_issues": self.schema_issues,
            "quality_issues": self.quality_issues,
            "status": self.status.value,
        }


@dataclass
class DataQualitySummary:
    """Aggregated data quality summary for a calibration run."""
    
    num_snapshots: int = 0
    num_schema_failures: int = 0
    num_low_quality_snapshots: int = 0
    overall_non_null_core_fraction: float = 0.0
    total_rows: int = 0
    status: DataQualityStatus = DataQualityStatus.OK
    issues: List[str] = field(default_factory=list)
    snapshot_reports: List[SnapshotQualityReport] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_snapshots": self.num_snapshots,
            "num_schema_failures": self.num_schema_failures,
            "num_low_quality_snapshots": self.num_low_quality_snapshots,
            "overall_non_null_core_fraction": round(self.overall_non_null_core_fraction, 4),
            "total_rows": self.total_rows,
            "status": self.status.value,
            "issues": self.issues,
        }


def _check_dtype_compatibility(actual_dtype, expected_type: str) -> bool:
    """Check if a column dtype matches expected type category."""
    dtype_str = str(actual_dtype).lower()
    
    if expected_type == "datetime64":
        return "datetime" in dtype_str or "timestamp" in dtype_str
    elif expected_type == "string":
        return "object" in dtype_str or "string" in dtype_str or "category" in dtype_str
    elif expected_type == "int64":
        return np.issubdtype(actual_dtype, np.integer) or np.issubdtype(actual_dtype, np.floating)
    elif expected_type == "float64":
        return np.issubdtype(actual_dtype, np.floating) or np.issubdtype(actual_dtype, np.integer)
    
    return True


def validate_snapshot_schema(
    df: pd.DataFrame,
    filename: str = "",
) -> List[str]:
    """
    Validate that a DataFrame has the expected schema for harvested snapshots.
    
    Returns a list of problems. Empty list means schema is OK.
    """
    issues: List[str] = []
    
    if df.empty:
        return ["DataFrame is empty"]
    
    actual_columns = set(df.columns)
    required_columns = set(REQUIRED_COLUMNS.keys())
    
    missing = required_columns - actual_columns
    if missing:
        issues.append(f"Missing columns: {sorted(missing)}")
    
    for col, expected_type in REQUIRED_COLUMNS.items():
        if col in df.columns:
            if not _check_dtype_compatibility(df[col].dtype, expected_type):
                issues.append(f"Column '{col}' has type {df[col].dtype}, expected {expected_type}")
    
    return issues


def assess_snapshot_quality(
    df: pd.DataFrame,
    filename: str = "",
    core_fraction_threshold: float = QUALITY_THRESHOLDS["core_field_fraction"],
) -> SnapshotQualityReport:
    """
    Assess the quality of a snapshot DataFrame.
    
    Checks for:
    - Non-zero row count
    - Fraction of rows with non-null core fields
    """
    report = SnapshotQualityReport(filename=filename)
    
    report.schema_issues = validate_snapshot_schema(df, filename)
    
    if df.empty:
        report.quality_issues.append("Snapshot contains zero rows")
        return report
    
    report.total_rows = len(df)
    
    present_core_fields = [f for f in CORE_FIELDS if f in df.columns]
    
    if not present_core_fields:
        report.non_null_core_fraction = 0.0
        report.quality_issues.append("No core fields present in snapshot")
        return report
    
    non_null_mask = df[present_core_fields].notna().all(axis=1)
    report.non_null_core_fraction = float(non_null_mask.mean())
    
    if report.non_null_core_fraction < core_fraction_threshold:
        missing_pct = (1 - report.non_null_core_fraction) * 100
        report.quality_issues.append(
            f"More than {missing_pct:.1f}% of rows are missing core fields"
        )
    
    return report


def aggregate_quality_reports(
    reports: List[SnapshotQualityReport],
    degraded_threshold: float = QUALITY_THRESHOLDS["degraded_snapshot_fraction"],
) -> DataQualitySummary:
    """
    Aggregate multiple snapshot quality reports into a summary.
    
    Determines overall status based on:
    - Any schema failures → FAILED
    - Too many low-quality snapshots → DEGRADED
    - Otherwise → OK
    """
    summary = DataQualitySummary()
    summary.num_snapshots = len(reports)
    summary.snapshot_reports = reports
    
    if not reports:
        summary.status = DataQualityStatus.OK
        summary.issues.append("No snapshots to evaluate")
        return summary
    
    summary.num_schema_failures = sum(1 for r in reports if not r.is_schema_valid)
    
    if summary.num_schema_failures > 0:
        summary.status = DataQualityStatus.FAILED
        failed_files = [r.filename for r in reports if not r.is_schema_valid][:5]
        schema_issues = []
        for r in reports:
            if r.schema_issues:
                schema_issues.extend(r.schema_issues[:3])
        summary.issues.append(
            f"{summary.num_schema_failures} snapshots failed schema validation"
        )
        if schema_issues:
            summary.issues.append(f"Schema issues: {'; '.join(schema_issues[:5])}")
        if failed_files:
            summary.issues.append(f"Failed files: {', '.join(failed_files)}")
        return summary
    
    summary.num_low_quality_snapshots = sum(1 for r in reports if not r.is_quality_ok)
    
    total_rows = sum(r.total_rows for r in reports)
    summary.total_rows = total_rows
    
    if total_rows > 0:
        weighted_fraction = sum(
            r.non_null_core_fraction * r.total_rows for r in reports
        ) / total_rows
        summary.overall_non_null_core_fraction = weighted_fraction
    else:
        summary.overall_non_null_core_fraction = 0.0
    
    low_quality_fraction = summary.num_low_quality_snapshots / len(reports)
    
    if low_quality_fraction > degraded_threshold:
        summary.status = DataQualityStatus.DEGRADED
        summary.issues.append(
            f"{summary.num_low_quality_snapshots} of {len(reports)} snapshots "
            f"({low_quality_fraction:.1%}) have quality issues"
        )
    elif summary.overall_non_null_core_fraction < QUALITY_THRESHOLDS["core_field_fraction"]:
        summary.status = DataQualityStatus.DEGRADED
        summary.issues.append(
            f"Overall core-field completeness is {summary.overall_non_null_core_fraction:.1%}, "
            f"below {QUALITY_THRESHOLDS['core_field_fraction']:.1%} threshold"
        )
    else:
        summary.status = DataQualityStatus.OK
    
    return summary


def format_quality_summary_for_display(summary: DataQualitySummary) -> str:
    """
    Format a quality summary into a human-readable string.
    """
    lines = [
        f"Data health:",
        f"  - Snapshots checked: {summary.num_snapshots}",
        f"  - Schema issues: {summary.num_schema_failures}",
        f"  - Low-quality snapshots: {summary.num_low_quality_snapshots}",
        f"  - Overall core-field completeness: {summary.overall_non_null_core_fraction:.0%}",
        f"  - Status: {summary.status.value.upper()}",
    ]
    
    if summary.issues:
        lines.append("  - Issues:")
        for issue in summary.issues[:5]:
            lines.append(f"      {issue}")
    
    return "\n".join(lines)
