"""
Drawdown Store Module.

Persists daily drawdown state to a file so it survives restarts.
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional


@dataclass
class DailyDrawdownState:
    """Daily drawdown tracking state."""
    date: date
    max_equity_usd: float
    
    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "date": self.date.isoformat(),
            "max_equity_usd": self.max_equity_usd,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "DailyDrawdownState":
        """Create from dict."""
        return cls(
            date=date.fromisoformat(d["date"]),
            max_equity_usd=float(d["max_equity_usd"]),
        )


# Default path for state file
DEFAULT_STATE_FILE = Path("data/drawdown_state.json")


def _get_state_file_path() -> Path:
    """Get the path to the state file, creating parent directories if needed."""
    state_file = DEFAULT_STATE_FILE
    state_file.parent.mkdir(parents=True, exist_ok=True)
    return state_file


def load_daily_drawdown_state() -> Optional[DailyDrawdownState]:
    """
    Load daily drawdown state from file.
    
    Returns:
        DailyDrawdownState if file exists and is valid, None otherwise.
    """
    state_file = _get_state_file_path()
    
    if not state_file.exists():
        return None
    
    try:
        with open(state_file, "r") as f:
            data = json.load(f)
        
        if not data:
            return None
        
        return DailyDrawdownState.from_dict(data)
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        print(f"[DRAWDOWN STORE] Warning: Could not load state: {e}")
        return None


def save_daily_drawdown_state(state: DailyDrawdownState) -> bool:
    """
    Save daily drawdown state to file.
    Uses atomic write with temp file for safety.
    
    Returns:
        True if save was successful, False otherwise.
    """
    state_file = _get_state_file_path()
    
    try:
        # Write to temp file first for atomicity
        fd, temp_path = tempfile.mkstemp(
            dir=state_file.parent,
            prefix=".drawdown_state_",
            suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(state.to_dict(), f, indent=2)
            
            # Atomic rename
            os.replace(temp_path, state_file)
            return True
        except Exception:
            # Clean up temp file on error
            try:
                os.unlink(temp_path)
            except OSError:
                pass
            raise
    except Exception as e:
        print(f"[DRAWDOWN STORE] Error saving state: {e}")
        return False


def reset_daily_drawdown_state() -> bool:
    """
    Reset daily drawdown state by deleting the state file.
    
    Returns:
        True if reset was successful (or file didn't exist), False otherwise.
    """
    state_file = _get_state_file_path()
    
    try:
        if state_file.exists():
            state_file.unlink()
        return True
    except Exception as e:
        print(f"[DRAWDOWN STORE] Error resetting state: {e}")
        return False


def get_daily_drawdown_status() -> dict:
    """
    Get current daily drawdown status for API.
    
    Returns:
        Dict with current state info.
    """
    state = load_daily_drawdown_state()
    today = datetime.now(timezone.utc).date()
    
    if state is None:
        return {
            "has_state": False,
            "date": None,
            "max_equity_usd": None,
            "is_current_day": False,
        }
    
    return {
        "has_state": True,
        "date": state.date.isoformat(),
        "max_equity_usd": state.max_equity_usd,
        "is_current_day": state.date == today,
    }

