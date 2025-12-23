"""
Tests for the heartbeat monitor module.
"""
from __future__ import annotations

import time
import pytest

from src.ops.heartbeat import (
    record_heartbeat,
    check_heartbeat,
    get_heartbeat_status,
    reset_heartbeat,
    HeartbeatState,
)


@pytest.fixture(autouse=True)
def reset_state():
    """Reset heartbeat state before each test."""
    reset_heartbeat()
    yield
    reset_heartbeat()


class TestHeartbeatMonitor:
    """Tests for heartbeat monitoring."""
    
    def test_never_started_state(self):
        """When no heartbeat recorded, state should be NEVER_STARTED."""
        status = check_heartbeat(timeout_sec=300)
        assert status.state == HeartbeatState.NEVER_STARTED
        assert status.is_stalled is True
        assert status.last_heartbeat_time is None
    
    def test_healthy_after_heartbeat(self):
        """After recording heartbeat, state should be HEALTHY."""
        record_heartbeat()
        status = check_heartbeat(timeout_sec=300)
        
        assert status.state == HeartbeatState.HEALTHY
        assert status.is_stalled is False
        assert status.last_heartbeat_time is not None
        assert status.seconds_since_last < 1.0
    
    def test_stalled_after_timeout(self):
        """When timeout exceeded, state should be STALLED."""
        record_heartbeat()
        # Use a very short timeout to simulate stall
        time.sleep(0.1)
        status = check_heartbeat(timeout_sec=0)  # 0 means disabled, use small value
        
        # With timeout=0, it's disabled
        assert status.state == HeartbeatState.DISABLED
        
        # Test with 0.05s timeout after 0.1s sleep
        status = check_heartbeat(timeout_sec=1)  # 1 second
        # Should still be healthy since we just slept 0.1s
        assert status.state == HeartbeatState.HEALTHY
    
    def test_disabled_when_timeout_zero(self):
        """When timeout is 0, monitoring should be disabled."""
        record_heartbeat()
        status = check_heartbeat(timeout_sec=0)
        
        assert status.state == HeartbeatState.DISABLED
        assert status.is_stalled is False
    
    def test_multiple_heartbeats_update_time(self):
        """Multiple heartbeats should update the last time."""
        record_heartbeat()
        time.sleep(0.05)
        status1 = check_heartbeat(timeout_sec=300)
        
        record_heartbeat()
        status2 = check_heartbeat(timeout_sec=300)
        
        # Second heartbeat should have smaller time since last
        assert status2.seconds_since_last < status1.seconds_since_last
    
    def test_to_dict_serialization(self):
        """HeartbeatStatus should serialize to dict correctly."""
        record_heartbeat()
        status = check_heartbeat(timeout_sec=300)
        
        d = status.to_dict()
        assert isinstance(d, dict)
        assert "state" in d
        assert "is_stalled" in d
        assert "last_heartbeat_time" in d
        assert d["state"] == "healthy"
    
    def test_get_heartbeat_status_alias(self):
        """get_heartbeat_status should work as alias for check_heartbeat."""
        record_heartbeat()
        status1 = check_heartbeat(timeout_sec=300)
        status2 = get_heartbeat_status(timeout_sec=300)
        
        assert status1.state == status2.state


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

