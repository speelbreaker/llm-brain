"""
Unit tests for harvester data health and schema validation module.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone

from src.harvester.health import (
    REQUIRED_COLUMNS,
    CORE_FIELDS,
    SnapshotQualityReport,
    DataQualityStatus,
    DataQualitySummary,
    validate_snapshot_schema,
    assess_snapshot_quality,
    aggregate_quality_reports,
)


class TestValidateSnapshotSchema:
    """Tests for validate_snapshot_schema function."""
    
    def test_empty_dataframe_returns_issue(self):
        df = pd.DataFrame()
        issues = validate_snapshot_schema(df)
        assert len(issues) == 1
        assert "empty" in issues[0].lower()
    
    def test_all_required_columns_present_returns_empty(self):
        df = pd.DataFrame({
            "harvest_time": [pd.Timestamp.now(tz='UTC')],
            "instrument_name": ["BTC-31DEC25-100000-C"],
            "underlying": ["BTC"],
            "expiry": ["2025-12-31"],
            "expiry_timestamp": [1735603200000],
            "option_type": ["C"],
            "strike": [100000.0],
            "underlying_price": [50000.0],
            "mark_price": [0.05],
            "best_bid_price": [0.04],
            "best_ask_price": [0.06],
            "mark_iv": [65.0],
            "open_interest": [100.0],
            "volume": [50.0],
            "greek_delta": [0.3],
            "greek_vega": [10.0],
        })
        issues = validate_snapshot_schema(df)
        assert len(issues) == 0
    
    def test_missing_columns_reported(self):
        df = pd.DataFrame({
            "harvest_time": [pd.Timestamp.now(tz='UTC')],
            "instrument_name": ["BTC-31DEC25-100000-C"],
            "underlying": ["BTC"],
        })
        issues = validate_snapshot_schema(df)
        assert len(issues) >= 1
        assert any("missing" in issue.lower() for issue in issues)
    
    def test_reports_specific_missing_columns(self):
        df = pd.DataFrame({
            "harvest_time": [pd.Timestamp.now(tz='UTC')],
            "instrument_name": ["BTC-31DEC25-100000-C"],
        })
        issues = validate_snapshot_schema(df)
        missing_issue = [i for i in issues if "missing" in i.lower()][0]
        assert "strike" in missing_issue or "Missing columns" in missing_issue


class TestAssessSnapshotQuality:
    """Tests for assess_snapshot_quality function."""
    
    def test_empty_dataframe_reports_zero_rows(self):
        df = pd.DataFrame()
        report = assess_snapshot_quality(df, filename="test.parquet")
        assert report.total_rows == 0
        assert any("zero rows" in issue.lower() for issue in report.quality_issues)
        assert report.status == DataQualityStatus.FAILED
    
    def test_high_quality_data_reports_ok(self):
        df = pd.DataFrame({
            "harvest_time": [pd.Timestamp.now(tz='UTC')] * 100,
            "instrument_name": ["BTC-31DEC25-100000-C"] * 100,
            "underlying": ["BTC"] * 100,
            "expiry": ["2025-12-31"] * 100,
            "expiry_timestamp": [1735603200000] * 100,
            "option_type": ["C"] * 100,
            "strike": [100000.0] * 100,
            "underlying_price": [50000.0] * 100,
            "mark_price": [0.05] * 100,
            "best_bid_price": [0.04] * 100,
            "best_ask_price": [0.06] * 100,
            "mark_iv": [65.0] * 100,
            "open_interest": [100.0] * 100,
            "volume": [50.0] * 100,
            "greek_delta": [0.3] * 100,
            "greek_vega": [10.0] * 100,
        })
        report = assess_snapshot_quality(df, filename="test.parquet")
        assert report.total_rows == 100
        assert report.non_null_core_fraction == 1.0
        assert report.is_quality_ok
        assert report.status == DataQualityStatus.OK
    
    def test_low_quality_data_reports_issue(self):
        df = pd.DataFrame({
            "harvest_time": [pd.Timestamp.now(tz='UTC')] * 100,
            "instrument_name": ["BTC-31DEC25-100000-C"] * 100,
            "underlying": ["BTC"] * 100,
            "expiry": ["2025-12-31"] * 100,
            "expiry_timestamp": [1735603200000] * 100,
            "option_type": ["C"] * 50 + [None] * 50,
            "strike": [100000.0] * 100,
            "underlying_price": [50000.0] * 100,
            "mark_price": [None] * 80 + [0.05] * 20,
            "best_bid_price": [0.04] * 100,
            "best_ask_price": [0.06] * 100,
            "mark_iv": [65.0] * 100,
            "open_interest": [100.0] * 100,
            "volume": [50.0] * 100,
            "greek_delta": [0.3] * 100,
            "greek_vega": [10.0] * 100,
        })
        report = assess_snapshot_quality(df, filename="test.parquet")
        assert report.total_rows == 100
        assert report.non_null_core_fraction < 0.9
        assert not report.is_quality_ok
        assert report.status == DataQualityStatus.DEGRADED
    
    def test_filename_preserved(self):
        df = pd.DataFrame({"underlying_price": [50000.0]})
        report = assess_snapshot_quality(df, filename="my_snapshot.parquet")
        assert report.filename == "my_snapshot.parquet"


class TestSnapshotQualityReport:
    """Tests for SnapshotQualityReport dataclass."""
    
    def test_is_schema_valid_true_when_no_issues(self):
        report = SnapshotQualityReport(schema_issues=[])
        assert report.is_schema_valid
    
    def test_is_schema_valid_false_when_issues(self):
        report = SnapshotQualityReport(schema_issues=["Missing column X"])
        assert not report.is_schema_valid
    
    def test_is_quality_ok_true_when_no_issues(self):
        report = SnapshotQualityReport(quality_issues=[])
        assert report.is_quality_ok
    
    def test_is_quality_ok_false_when_issues(self):
        report = SnapshotQualityReport(quality_issues=["Low completeness"])
        assert not report.is_quality_ok
    
    def test_to_dict_returns_all_fields(self):
        report = SnapshotQualityReport(
            filename="test.parquet",
            total_rows=100,
            non_null_core_fraction=0.95,
            schema_issues=[],
            quality_issues=[],
        )
        d = report.to_dict()
        assert d["filename"] == "test.parquet"
        assert d["total_rows"] == 100
        assert d["status"] == "ok"


class TestAggregateQualityReports:
    """Tests for aggregate_quality_reports function."""
    
    def test_empty_reports_returns_ok_status(self):
        summary = aggregate_quality_reports([])
        assert summary.num_snapshots == 0
        assert summary.status == DataQualityStatus.OK
    
    def test_all_ok_reports_returns_ok(self):
        reports = [
            SnapshotQualityReport(filename="a.parquet", total_rows=100, non_null_core_fraction=0.99),
            SnapshotQualityReport(filename="b.parquet", total_rows=100, non_null_core_fraction=0.98),
        ]
        summary = aggregate_quality_reports(reports)
        assert summary.num_snapshots == 2
        assert summary.num_schema_failures == 0
        assert summary.num_low_quality_snapshots == 0
        assert summary.status == DataQualityStatus.OK
    
    def test_schema_failures_return_failed(self):
        reports = [
            SnapshotQualityReport(filename="a.parquet", schema_issues=["Missing column"]),
            SnapshotQualityReport(filename="b.parquet", total_rows=100, non_null_core_fraction=0.99),
        ]
        summary = aggregate_quality_reports(reports)
        assert summary.num_schema_failures == 1
        assert summary.status == DataQualityStatus.FAILED
    
    def test_many_low_quality_returns_degraded(self):
        reports = [
            SnapshotQualityReport(
                filename=f"{i}.parquet",
                total_rows=100,
                non_null_core_fraction=0.7,
                quality_issues=["Low completeness"],
            ) for i in range(10)
        ]
        summary = aggregate_quality_reports(reports)
        assert summary.num_low_quality_snapshots == 10
        assert summary.status == DataQualityStatus.DEGRADED
    
    def test_total_rows_aggregated(self):
        reports = [
            SnapshotQualityReport(filename="a.parquet", total_rows=100, non_null_core_fraction=0.99),
            SnapshotQualityReport(filename="b.parquet", total_rows=200, non_null_core_fraction=0.95),
        ]
        summary = aggregate_quality_reports(reports)
        assert summary.total_rows == 300
    
    def test_weighted_fraction_calculated(self):
        reports = [
            SnapshotQualityReport(filename="a.parquet", total_rows=100, non_null_core_fraction=1.0),
            SnapshotQualityReport(filename="b.parquet", total_rows=100, non_null_core_fraction=0.8),
        ]
        summary = aggregate_quality_reports(reports)
        assert 0.89 < summary.overall_non_null_core_fraction < 0.91
    
    def test_to_dict_returns_all_fields(self):
        summary = DataQualitySummary(
            num_snapshots=10,
            num_schema_failures=0,
            num_low_quality_snapshots=2,
            overall_non_null_core_fraction=0.94,
            total_rows=1000,
            status=DataQualityStatus.OK,
            issues=["Some warning"],
        )
        d = summary.to_dict()
        assert d["num_snapshots"] == 10
        assert d["status"] == "ok"
        assert "Some warning" in d["issues"]


class TestDataQualityStatus:
    """Tests for DataQualityStatus enum."""
    
    def test_enum_values(self):
        assert DataQualityStatus.OK.value == "ok"
        assert DataQualityStatus.DEGRADED.value == "degraded"
        assert DataQualityStatus.FAILED.value == "failed"
    
    def test_is_string_enum(self):
        assert str(DataQualityStatus.OK) == "DataQualityStatus.OK"
        assert DataQualityStatus.OK.value == "ok"
