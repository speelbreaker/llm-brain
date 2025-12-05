"""
In-memory store for recent agent decisions.
Provides a thread-safe buffer of the last N decisions for the dashboard.
"""
from __future__ import annotations

import threading
from collections import deque
from typing import Any, Dict, List, Optional
from datetime import datetime


class DecisionsStore:
    """Thread-safe store for recent agent decisions."""
    
    def __init__(self, max_size: int = 50):
        self._decisions: deque = deque(maxlen=max_size)
        self._lock = threading.Lock()
        self._last_update: Optional[datetime] = None
    
    def add(self, decision: Dict[str, Any]) -> None:
        """Add a new decision to the buffer (newest first)."""
        with self._lock:
            self._decisions.appendleft(decision)
            self._last_update = datetime.utcnow()
    
    def get_all(self) -> List[Dict[str, Any]]:
        """Get all decisions (newest first)."""
        with self._lock:
            return list(self._decisions)
    
    def get_latest(self) -> Optional[Dict[str, Any]]:
        """Get the most recent decision."""
        with self._lock:
            if self._decisions:
                return self._decisions[0]
            return None
    
    def get_last_update(self) -> Optional[datetime]:
        """Get the timestamp of the last update."""
        with self._lock:
            return self._last_update
    
    def clear(self) -> None:
        """Clear all decisions."""
        with self._lock:
            self._decisions.clear()
            self._last_update = None


decisions_store = DecisionsStore(max_size=50)
