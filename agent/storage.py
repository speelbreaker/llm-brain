"""
Storage layer for the Telegram Code Review Agent.

Uses SQLite for persistent storage of:
- Review history
- Metadata (last reviewed commit, etc.)
- File snapshots (for fallback change detection)
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.config import settings


@dataclass
class ReviewRecord:
    """A stored review record."""
    id: int
    created_at: str
    initiator_telegram_id: str
    target_type: str
    target_ref: str
    git_head: Optional[str]
    change_detector_mode: str
    overall_severity: str
    summary_md: str
    issues_json: str
    next_steps_json: str
    diff_summary_json: Optional[str] = None
    
    @property
    def issues(self) -> List[Dict[str, Any]]:
        """Parse issues from JSON."""
        try:
            return json.loads(self.issues_json) if self.issues_json else []
        except json.JSONDecodeError:
            return []
    
    @property
    def next_steps(self) -> List[str]:
        """Parse next steps from JSON."""
        try:
            return json.loads(self.next_steps_json) if self.next_steps_json else []
        except json.JSONDecodeError:
            return []
    
    @property
    def diff_summary(self) -> List[Dict[str, Any]]:
        """Parse diff summary from JSON."""
        try:
            return json.loads(self.diff_summary_json) if self.diff_summary_json else []
        except json.JSONDecodeError:
            return []


def _get_db_path() -> Path:
    """Get database path, creating parent directories if needed."""
    if settings:
        db_path = settings.db_path
    else:
        db_path = Path("data/agent_data.db")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return db_path


def _get_conn() -> sqlite3.Connection:
    """Get a database connection."""
    return sqlite3.connect(_get_db_path())


def init_db() -> None:
    """Initialize the database schema."""
    conn = _get_conn()
    cur = conn.cursor()
    
    cur.execute("""
        CREATE TABLE IF NOT EXISTS reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            initiator_telegram_id TEXT,
            target_type TEXT NOT NULL,
            target_ref TEXT,
            git_head TEXT,
            change_detector_mode TEXT,
            overall_severity TEXT,
            summary_md TEXT,
            issues_json TEXT,
            next_steps_json TEXT,
            diff_summary_json TEXT
        )
    """)
    
    cur.execute("""
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    """)
    
    cur.execute("""
        CREATE TABLE IF NOT EXISTS snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            label TEXT NOT NULL,
            data_json TEXT
        )
    """)
    
    conn.commit()
    conn.close()


def set_meta(key: str, value: str) -> None:
    """Set a metadata value."""
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, value))
    conn.commit()
    conn.close()


def get_meta(key: str, default: Optional[str] = None) -> Optional[str]:
    """Get a metadata value."""
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute("SELECT value FROM meta WHERE key = ?", (key,))
    row = cur.fetchone()
    conn.close()
    return row[0] if row else default


def save_review(
    initiator_id: int,
    target_type: str,
    target_ref: str,
    git_head: Optional[str],
    change_detector_mode: str,
    overall_severity: str,
    summary_md: str,
    issues: List[Dict[str, Any]],
    next_steps: List[str],
    diff_summary: Optional[List[Dict[str, Any]]] = None,
) -> int:
    """Save a review record and return its ID."""
    conn = _get_conn()
    cur = conn.cursor()
    
    now = datetime.now(timezone.utc).isoformat()
    
    cur.execute("""
        INSERT INTO reviews (
            created_at, initiator_telegram_id, target_type, target_ref,
            git_head, change_detector_mode, overall_severity,
            summary_md, issues_json, next_steps_json, diff_summary_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        now,
        str(initiator_id),
        target_type,
        target_ref,
        git_head,
        change_detector_mode,
        overall_severity,
        summary_md,
        json.dumps(issues),
        json.dumps(next_steps),
        json.dumps(diff_summary) if diff_summary else None,
    ))
    
    review_id = cur.lastrowid or 0
    conn.commit()
    conn.close()
    
    return review_id


def get_last_review() -> Optional[ReviewRecord]:
    """Get the most recent review."""
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT id, created_at, initiator_telegram_id, target_type, target_ref,
               git_head, change_detector_mode, overall_severity,
               summary_md, issues_json, next_steps_json, diff_summary_json
        FROM reviews
        ORDER BY id DESC
        LIMIT 1
    """)
    row = cur.fetchone()
    conn.close()
    
    if not row:
        return None
    
    return ReviewRecord(
        id=row[0],
        created_at=row[1],
        initiator_telegram_id=row[2],
        target_type=row[3],
        target_ref=row[4],
        git_head=row[5],
        change_detector_mode=row[6],
        overall_severity=row[7],
        summary_md=row[8],
        issues_json=row[9],
        next_steps_json=row[10],
        diff_summary_json=row[11],
    )


def get_review_count() -> int:
    """Get total number of reviews."""
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM reviews")
    count = cur.fetchone()[0]
    conn.close()
    return count


def save_snapshot(label: str, data: Dict[str, Any]) -> int:
    """Save a file snapshot."""
    conn = _get_conn()
    cur = conn.cursor()
    
    now = datetime.now(timezone.utc).isoformat()
    
    cur.execute("""
        INSERT INTO snapshots (created_at, label, data_json)
        VALUES (?, ?, ?)
    """, (now, label, json.dumps(data)))
    
    snapshot_id = cur.lastrowid or 0
    conn.commit()
    conn.close()
    
    return snapshot_id


def get_last_snapshot(label: str) -> Optional[Dict[str, Any]]:
    """Get the most recent snapshot for a label."""
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT data_json FROM snapshots
        WHERE label = ?
        ORDER BY id DESC
        LIMIT 1
    """, (label,))
    row = cur.fetchone()
    conn.close()
    
    if not row or not row[0]:
        return None
    
    try:
        return json.loads(row[0])
    except json.JSONDecodeError:
        return None
