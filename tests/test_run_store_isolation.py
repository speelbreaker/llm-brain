"""
Tests for run_store isolation via BACKTEST_RUNS_DIR environment variable.

These tests verify that:
1. The env override works correctly
2. Files are created in the temp directory, not in data/backtests
"""
import os
from pathlib import Path

import pytest

from src.backtest import run_store


class TestBacktestRunsIsolation:
    """Tests for BACKTEST_RUNS_DIR environment variable override."""

    def test_env_override_changes_backtests_dir(self, isolate_backtest_storage: Path):
        """BACKTEST_RUNS_DIR should override the default backtests directory."""
        actual_dir = run_store._get_backtests_dir()
        assert actual_dir == isolate_backtest_storage
        assert str(actual_dir) != "data/backtests"

    def test_env_override_changes_index_file_path(self, isolate_backtest_storage: Path):
        """Index file should be created in the overridden directory."""
        index_file = run_store._get_index_file()
        assert index_file.parent == isolate_backtest_storage
        assert index_file.name == "index.jsonl"

    def test_create_run_writes_to_temp_dir(self, isolate_backtest_storage: Path):
        """create_run should write files to the overridden temp directory."""
        config = {
            "underlying": "BTC",
            "start_date": "2024-01-01",
            "end_date": "2024-01-07",
        }
        
        result = run_store.create_run(config)
        
        run_dir = isolate_backtest_storage / result.run_id
        assert run_dir.exists(), "Run directory should exist in temp dir"
        
        result_file = run_dir / "result.json"
        assert result_file.exists(), "result.json should exist in temp dir"
        
        index_file = isolate_backtest_storage / "index.jsonl"
        assert index_file.exists(), "index.jsonl should exist in temp dir"
        
        production_dir = Path("data/backtests") / result.run_id
        assert not production_dir.exists(), "Run should NOT be created in production dir"

    def test_load_index_reads_from_temp_dir(self, isolate_backtest_storage: Path):
        """load_index should read from the overridden temp directory."""
        config = {
            "underlying": "ETH",
            "start_date": "2024-02-01",
            "end_date": "2024-02-07",
        }
        
        run_store.create_run(config)
        
        entries = run_store.load_index()
        
        assert len(entries) >= 1
        assert entries[0].underlying == "ETH"

    def test_default_dir_used_when_env_not_set(self, monkeypatch: pytest.MonkeyPatch):
        """When BACKTEST_RUNS_DIR is not set, default directory should be used."""
        monkeypatch.delenv("BACKTEST_RUNS_DIR", raising=False)
        
        actual_dir = run_store._get_backtests_dir()
        assert actual_dir == Path("data/backtests")
