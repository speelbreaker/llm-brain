"""
Pytest configuration and shared fixtures.

This module provides fixtures that ensure tests don't pollute the
production data directories (backtests, databases, etc.).
"""
import os
import tempfile
from pathlib import Path
from typing import Generator

import pytest


@pytest.fixture(autouse=True)
def isolate_backtest_storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Generator[Path, None, None]:
    """
    Automatically isolate backtest storage to a temp directory for all tests.
    
    This fixture:
    - Sets BACKTEST_RUNS_DIR to a per-test temp directory
    - Ensures no test writes to data/backtests/ or modifies index.jsonl
    - Cleans up automatically after each test
    
    Yields:
        Path to the isolated backtest directory
    """
    backtest_dir = tmp_path / "backtests"
    backtest_dir.mkdir(parents=True, exist_ok=True)
    
    monkeypatch.setenv("BACKTEST_RUNS_DIR", str(backtest_dir))
    
    yield backtest_dir


@pytest.fixture(autouse=True)
def isolate_test_database(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Generator[Path, None, None]:
    """
    Automatically isolate test databases to a temp directory.
    
    This prevents .test_db.sqlite and data/agent_data.db from being 
    created/modified in the repo.
    
    Yields:
        Path to the isolated database file
    """
    db_dir = tmp_path / "db"
    db_dir.mkdir(parents=True, exist_ok=True)
    
    agent_db_path = db_dir / "agent_data.db"
    monkeypatch.setenv("AGENT_DB_PATH", str(agent_db_path))
    
    test_db_path = db_dir / "test_db.sqlite"
    monkeypatch.setenv("TEST_DB_PATH", str(test_db_path))
    
    yield test_db_path
