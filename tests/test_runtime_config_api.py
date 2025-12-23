"""
Tests for the runtime config API endpoints.

Verifies that trade_mode can be read and updated via the API.
"""
from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock

from fastapi.testclient import TestClient

from src.config import Settings, TradingMode


@pytest.fixture
def test_settings():
    """Create test settings for API testing."""
    return Settings(
        deribit_env="testnet",
        deribit_base_url="https://test.deribit.com",
        mode="research",
        dry_run=True,
        kill_switch_enabled=False,
        trade_mode=TradingMode.NORMAL,
    )


@pytest.fixture
def client(test_settings):
    """Create test client with patched settings."""
    # We need to patch the settings in the routes_health module
    with patch("src.web.routes_health.settings", test_settings):
        # Import after patching
        from src.web.routes_health import router
        from fastapi import FastAPI
        
        app = FastAPI()
        app.include_router(router)
        
        yield TestClient(app)


class TestRuntimeConfigEndpoint:
    """Test runtime config GET and POST endpoints."""
    
    def test_get_runtime_config_includes_trade_mode(self, client, test_settings):
        """GET /api/system/runtime-config should include trade_mode."""
        response = client.get("/api/system/runtime-config")
        assert response.status_code == 200
        
        data = response.json()
        assert data["ok"] is True
        assert "trade_mode" in data
        assert data["trade_mode"] == "normal"
    
    def test_update_trade_mode_to_close_only(self, client, test_settings):
        """POST /api/system/runtime-config should update trade_mode to close_only."""
        response = client.post(
            "/api/system/runtime-config",
            json={"trade_mode": "close_only"}
        )
        assert response.status_code == 200
        
        data = response.json()
        assert data["ok"] is True
        assert "trade_mode" in data.get("updated", {})
        assert data["updated"]["trade_mode"] == "close_only"
        assert data["current"]["trade_mode"] == "close_only"
        
        # Verify settings object was updated
        assert test_settings.trade_mode == TradingMode.CLOSE_ONLY
    
    def test_update_trade_mode_to_halt(self, client, test_settings):
        """POST /api/system/runtime-config should update trade_mode to halt."""
        response = client.post(
            "/api/system/runtime-config",
            json={"trade_mode": "halt"}
        )
        assert response.status_code == 200
        
        data = response.json()
        assert data["ok"] is True
        assert data["current"]["trade_mode"] == "halt"
        assert test_settings.trade_mode == TradingMode.HALT
    
    def test_update_trade_mode_invalid_value(self, client, test_settings):
        """POST /api/system/runtime-config should reject invalid trade_mode."""
        response = client.post(
            "/api/system/runtime-config",
            json={"trade_mode": "invalid_mode"}
        )
        assert response.status_code == 400
        
        data = response.json()
        assert data["ok"] is False
        assert any("trade_mode" in err for err in data.get("errors", []))
    
    def test_update_trade_mode_back_to_normal(self, client, test_settings):
        """Should be able to update trade_mode back to normal."""
        # First set to halt
        client.post("/api/system/runtime-config", json={"trade_mode": "halt"})
        assert test_settings.trade_mode == TradingMode.HALT
        
        # Then back to normal
        response = client.post(
            "/api/system/runtime-config",
            json={"trade_mode": "normal"}
        )
        assert response.status_code == 200
        assert test_settings.trade_mode == TradingMode.NORMAL
    
    def test_update_multiple_fields_including_trade_mode(self, client, test_settings):
        """Should be able to update trade_mode along with other fields."""
        response = client.post(
            "/api/system/runtime-config",
            json={
                "trade_mode": "close_only",
                "kill_switch_enabled": True,
            }
        )
        assert response.status_code == 200
        
        data = response.json()
        assert data["ok"] is True
        assert data["current"]["trade_mode"] == "close_only"
        assert data["current"]["kill_switch_enabled"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
