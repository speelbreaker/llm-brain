"""
Persistent storage for backtest runs.

Stores backtest results in:
  data/backtests/<run_id>/result.json

Maintains an index file at:
  data/backtests/index.jsonl
"""
from __future__ import annotations

import json
import os
import threading
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

StatusType = Literal["queued", "running", "finished", "failed"]

BACKTESTS_DIR = Path("data/backtests")
INDEX_FILE = BACKTESTS_DIR / "index.jsonl"

_lock = threading.Lock()


@dataclass
class BacktestRunConfig:
    """Configuration snapshot for a backtest run."""
    underlying: str
    start_date: str
    end_date: str
    timeframe: str = "1h"
    decision_interval_bars: int = 24
    target_dte: int = 7
    target_delta: float = 0.25
    dte_tolerance: int = 2
    delta_tolerance: float = 0.05
    initial_position: float = 1.0
    exit_style: str = "both"
    pricing_mode: str = "synthetic_bs"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BacktestMetrics:
    """Performance metrics for a single exit style."""
    initial_equity: float = 0.0
    final_equity: float = 0.0
    net_profit_usd: float = 0.0
    net_profit_pct: float = 0.0
    hodl_profit_usd: float = 0.0
    hodl_profit_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    num_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BacktestRunResult:
    """Full result of a backtest run."""
    run_id: str
    created_at: str
    status: StatusType
    config: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    recent_steps: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    recent_chains: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    equity_curves: Dict[str, List[List[Any]]] = field(default_factory=dict)
    error: Optional[str] = None
    finished_at: Optional[str] = None
    duration_seconds: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "created_at": self.created_at,
            "status": self.status,
            "config": self.config,
            "metrics": self.metrics,
            "recent_steps": self.recent_steps,
            "recent_chains": self.recent_chains,
            "equity_curves": self.equity_curves,
            "error": self.error,
            "finished_at": self.finished_at,
            "duration_seconds": self.duration_seconds,
        }


@dataclass
class BacktestIndexEntry:
    """Summary entry for the index file."""
    run_id: str
    created_at: str
    underlying: str
    start_date: str
    end_date: str
    status: StatusType
    primary_exit_style: str = "tp_and_roll"
    net_profit_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    num_trades: int = 0
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if d["error"] is None:
            del d["error"]
        return d


def generate_run_id(underlying: str = "BTC") -> str:
    """Generate a unique run ID with timestamp prefix."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    short_uuid = uuid.uuid4().hex[:8]
    return f"{ts}_{underlying}_{short_uuid}"


def get_run_dir(run_id: str) -> Path:
    """Get the directory path for a run."""
    return BACKTESTS_DIR / run_id


def get_result_path(run_id: str) -> Path:
    """Get the result.json path for a run."""
    return get_run_dir(run_id) / "result.json"


def ensure_backtests_dir() -> None:
    """Ensure the backtests directory exists."""
    BACKTESTS_DIR.mkdir(parents=True, exist_ok=True)


def create_run(config: Dict[str, Any]) -> BacktestRunResult:
    """
    Create a new backtest run in queued state.
    
    Args:
        config: Backtest configuration dict
        
    Returns:
        BacktestRunResult with queued status
    """
    ensure_backtests_dir()
    
    underlying = config.get("underlying", "BTC")
    run_id = generate_run_id(underlying)
    created_at = datetime.now(timezone.utc).isoformat()
    
    result = BacktestRunResult(
        run_id=run_id,
        created_at=created_at,
        status="queued",
        config=config,
    )
    
    run_dir = get_run_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    
    _save_result(result)
    
    index_entry = BacktestIndexEntry(
        run_id=run_id,
        created_at=created_at,
        underlying=underlying,
        start_date=config.get("start_date", config.get("start", "")),
        end_date=config.get("end_date", config.get("end", "")),
        status="queued",
        primary_exit_style=config.get("exit_style", "tp_and_roll"),
    )
    _append_index_entry(index_entry)
    
    return result


def update_run_status(run_id: str, status: StatusType, error: Optional[str] = None) -> None:
    """
    Update the status of a run.
    
    Args:
        run_id: Run identifier
        status: New status
        error: Optional error message for failed runs
    """
    result = load_result(run_id)
    if result is None:
        return
    
    result.status = status
    if error:
        result.error = error
    if status in ("finished", "failed"):
        result.finished_at = datetime.now(timezone.utc).isoformat()
        if result.created_at:
            try:
                created = datetime.fromisoformat(result.created_at.replace("Z", "+00:00"))
                finished = datetime.fromisoformat(result.finished_at.replace("Z", "+00:00"))
                result.duration_seconds = (finished - created).total_seconds()
            except Exception:
                pass
    
    _save_result(result)
    _update_index_status(run_id, status, error)


def save_run_result(result: BacktestRunResult) -> None:
    """
    Save a complete backtest result.
    
    Args:
        result: The complete result to save
    """
    _save_result(result)
    
    primary_style = result.config.get("exit_style", "tp_and_roll")
    if primary_style == "both":
        primary_style = "tp_and_roll"
    
    metrics = result.metrics.get(primary_style, {})
    
    index_entry = BacktestIndexEntry(
        run_id=result.run_id,
        created_at=result.created_at,
        underlying=result.config.get("underlying", "BTC"),
        start_date=result.config.get("start_date", result.config.get("start", "")),
        end_date=result.config.get("end_date", result.config.get("end", "")),
        status=result.status,
        primary_exit_style=primary_style,
        net_profit_pct=metrics.get("net_profit_pct", 0.0),
        max_drawdown_pct=metrics.get("max_drawdown_pct", 0.0),
        sharpe_ratio=metrics.get("sharpe_ratio", 0.0),
        num_trades=metrics.get("num_trades", 0),
        error=result.error,
    )
    _update_index_entry(index_entry)


def load_result(run_id: str) -> Optional[BacktestRunResult]:
    """
    Load a backtest result from disk.
    
    Args:
        run_id: Run identifier
        
    Returns:
        BacktestRunResult or None if not found
    """
    result_path = get_result_path(run_id)
    if not result_path.exists():
        return None
    
    try:
        with open(result_path, "r") as f:
            data = json.load(f)
        return BacktestRunResult(
            run_id=data.get("run_id", run_id),
            created_at=data.get("created_at", ""),
            status=data.get("status", "queued"),
            config=data.get("config", {}),
            metrics=data.get("metrics", {}),
            recent_steps=data.get("recent_steps", {}),
            recent_chains=data.get("recent_chains", {}),
            equity_curves=data.get("equity_curves", {}),
            error=data.get("error"),
            finished_at=data.get("finished_at"),
            duration_seconds=data.get("duration_seconds"),
        )
    except Exception:
        return None


def load_index() -> List[BacktestIndexEntry]:
    """
    Load all index entries, sorted by created_at descending.
    
    Returns:
        List of BacktestIndexEntry objects
    """
    ensure_backtests_dir()
    
    if not INDEX_FILE.exists():
        return []
    
    entries_by_id: Dict[str, BacktestIndexEntry] = {}
    
    try:
        with open(INDEX_FILE, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    entry = BacktestIndexEntry(
                        run_id=data.get("run_id", ""),
                        created_at=data.get("created_at", ""),
                        underlying=data.get("underlying", ""),
                        start_date=data.get("start_date", ""),
                        end_date=data.get("end_date", ""),
                        status=data.get("status", "queued"),
                        primary_exit_style=data.get("primary_exit_style", "tp_and_roll"),
                        net_profit_pct=data.get("net_profit_pct", 0.0),
                        max_drawdown_pct=data.get("max_drawdown_pct", 0.0),
                        sharpe_ratio=data.get("sharpe_ratio", 0.0),
                        num_trades=data.get("num_trades", 0),
                        error=data.get("error"),
                    )
                    entries_by_id[entry.run_id] = entry
                except json.JSONDecodeError:
                    continue
    except Exception:
        return []
    
    entries = list(entries_by_id.values())
    entries.sort(key=lambda e: e.created_at, reverse=True)
    return entries


def delete_run(run_id: str) -> bool:
    """
    Delete a backtest run and its files.
    
    Args:
        run_id: Run identifier
        
    Returns:
        True if deleted, False otherwise
    """
    import shutil
    
    run_dir = get_run_dir(run_id)
    if run_dir.exists():
        shutil.rmtree(run_dir)
    
    _remove_from_index(run_id)
    return True


def _save_result(result: BacktestRunResult) -> None:
    """Save result to disk atomically."""
    run_dir = get_run_dir(result.run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    
    result_path = get_result_path(result.run_id)
    tmp_path = result_path.with_suffix(".tmp")
    
    with _lock:
        with open(tmp_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2, default=str)
        tmp_path.replace(result_path)


def _append_index_entry(entry: BacktestIndexEntry) -> None:
    """Append an entry to the index file."""
    ensure_backtests_dir()
    
    with _lock:
        with open(INDEX_FILE, "a") as f:
            f.write(json.dumps(entry.to_dict()) + "\n")


def _update_index_entry(entry: BacktestIndexEntry) -> None:
    """Update or append an entry in the index file."""
    with _lock:
        entries = load_index()
        found = False
        for i, e in enumerate(entries):
            if e.run_id == entry.run_id:
                entries[i] = entry
                found = True
                break
        if not found:
            entries.append(entry)
        
        _rewrite_index(entries)


def _update_index_status(run_id: str, status: StatusType, error: Optional[str] = None) -> None:
    """Update just the status field in the index."""
    with _lock:
        entries = load_index()
        for entry in entries:
            if entry.run_id == run_id:
                entry.status = status
                if error:
                    entry.error = error
                break
        _rewrite_index(entries)


def _remove_from_index(run_id: str) -> None:
    """Remove an entry from the index file."""
    with _lock:
        entries = load_index()
        entries = [e for e in entries if e.run_id != run_id]
        _rewrite_index(entries)


def _rewrite_index(entries: List[BacktestIndexEntry]) -> None:
    """Rewrite the entire index file."""
    ensure_backtests_dir()
    
    tmp_path = INDEX_FILE.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry.to_dict()) + "\n")
    tmp_path.replace(INDEX_FILE)
