"""
Heartbeat Monitor Module.

Tracks agent loop liveness and detects stalls.
The agent loop should call record_heartbeat() at the start of each iteration.
Other components can check heartbeat status to detect if the loop has stalled.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Optional


class HeartbeatState(str, Enum):
    """Heartbeat health states."""
    HEALTHY = "healthy"
    STALLED = "stalled"
    NEVER_STARTED = "never_started"
    DISABLED = "disabled"


@dataclass
class HeartbeatStatus:
    """Status of the heartbeat monitor."""
    state: HeartbeatState
    last_heartbeat_time: Optional[datetime]
    seconds_since_last: Optional[float]
    timeout_threshold_sec: int
    is_stalled: bool
    message: str
    
    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "state": self.state.value,
            "last_heartbeat_time": self.last_heartbeat_time.isoformat() if self.last_heartbeat_time else None,
            "seconds_since_last": round(self.seconds_since_last, 1) if self.seconds_since_last is not None else None,
            "timeout_threshold_sec": self.timeout_threshold_sec,
            "is_stalled": self.is_stalled,
            "message": self.message,
        }


# Module-level state
_last_heartbeat_time: Optional[float] = None
_iteration_count: int = 0


def record_heartbeat() -> None:
    """
    Record a heartbeat from the agent loop.
    Call this at the start of each agent loop iteration.
    """
    global _last_heartbeat_time, _iteration_count
    _last_heartbeat_time = time.time()
    _iteration_count += 1


def get_iteration_count() -> int:
    """Get the number of iterations since startup."""
    return _iteration_count


def check_heartbeat(timeout_sec: int = 300) -> HeartbeatStatus:
    """
    Check the heartbeat status.
    
    Args:
        timeout_sec: Threshold in seconds after which the agent is considered stalled.
                     0 means heartbeat monitoring is disabled.
    
    Returns:
        HeartbeatStatus with current state.
    """
    if timeout_sec <= 0:
        return HeartbeatStatus(
            state=HeartbeatState.DISABLED,
            last_heartbeat_time=None,
            seconds_since_last=None,
            timeout_threshold_sec=timeout_sec,
            is_stalled=False,
            message="Heartbeat monitoring disabled",
        )
    
    if _last_heartbeat_time is None:
        return HeartbeatStatus(
            state=HeartbeatState.NEVER_STARTED,
            last_heartbeat_time=None,
            seconds_since_last=None,
            timeout_threshold_sec=timeout_sec,
            is_stalled=True,  # Never started is also considered stalled
            message="Agent loop has never started",
        )
    
    now = time.time()
    seconds_since = now - _last_heartbeat_time
    last_dt = datetime.fromtimestamp(_last_heartbeat_time, tz=timezone.utc)
    
    if seconds_since > timeout_sec:
        return HeartbeatStatus(
            state=HeartbeatState.STALLED,
            last_heartbeat_time=last_dt,
            seconds_since_last=seconds_since,
            timeout_threshold_sec=timeout_sec,
            is_stalled=True,
            message=f"Agent loop stalled: {seconds_since:.0f}s since last heartbeat (threshold: {timeout_sec}s)",
        )
    
    return HeartbeatStatus(
        state=HeartbeatState.HEALTHY,
        last_heartbeat_time=last_dt,
        seconds_since_last=seconds_since,
        timeout_threshold_sec=timeout_sec,
        is_stalled=False,
        message=f"Healthy: {seconds_since:.0f}s since last heartbeat",
    )


def get_heartbeat_status(timeout_sec: int = 300) -> HeartbeatStatus:
    """
    Convenience function to get heartbeat status.
    Alias for check_heartbeat().
    """
    return check_heartbeat(timeout_sec)


def reset_heartbeat() -> None:
    """
    Reset heartbeat state. Useful for testing.
    """
    global _last_heartbeat_time, _iteration_count
    _last_heartbeat_time = None
    _iteration_count = 0

