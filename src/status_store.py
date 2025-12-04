"""
In-memory status store for sharing agent state between threads.
Thread-safe storage for the latest agent decision snapshot.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Dict, Optional


@dataclass
class StatusStore:
    """Thread-safe store for the latest agent status snapshot."""
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)
    _data: Optional[Dict[str, Any]] = field(default=None, init=False, repr=False)

    def update(self, snapshot: Dict[str, Any]) -> None:
        """Update the stored snapshot (thread-safe)."""
        with self._lock:
            self._data = snapshot

    def get(self) -> Dict[str, Any]:
        """Get the current snapshot (thread-safe). Returns a copy."""
        with self._lock:
            if self._data is not None:
                return self._data.copy()
            return {
                "status": "starting",
                "message": "Agent has not produced a status snapshot yet."
            }


status_store = StatusStore()
