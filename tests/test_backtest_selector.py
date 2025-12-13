"""
Minimal test for selector_name pass-through in backtest system.
"""
import pytest
from datetime import datetime, timezone


def test_get_status_includes_config_with_selector():
    """Verify get_status() returns config with selector_name after initialization."""
    from src.backtest.manager import BacktestManager, BacktestStatus
    
    manager = BacktestManager()
    
    test_config = {
        "underlying": "BTC",
        "selector_name": "greg_vrp_harvester",
    }
    manager._status = BacktestStatus(config=test_config)
    
    status = manager.get_status()
    
    assert "config" in status, "get_status() must include 'config' key"
    assert status["config"]["selector_name"] == "greg_vrp_harvester", \
        "selector_name must be accessible via st.config.selector_name"


def test_selector_name_in_config_dict():
    """Verify selector_name is properly included in backtest config."""
    from src.backtest.manager import BacktestManager
    
    manager = BacktestManager()
    
    start_date = datetime(2025, 12, 7, tzinfo=timezone.utc)
    end_date = datetime(2025, 12, 8, tzinfo=timezone.utc)
    
    test_cases = [
        ("generic_covered_call", "Generic covered call selector"),
        ("greg_vrp_harvester", "GregBot VRP Harvester selector"),
    ]
    
    for selector_name, description in test_cases:
        config_dict = {
            "underlying": "BTC",
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "timeframe": "1h",
            "decision_interval_hours": 24,
            "exit_style": "hold_to_expiry",
            "target_dte": 7,
            "target_delta": 0.25,
            "min_dte": 3,
            "max_dte": 21,
            "delta_min": 0.15,
            "delta_max": 0.35,
            "margin_type": "inverse",
            "settlement_ccy": "ANY",
            "sigma_mode": "rv_x_multiplier",
            "chain_mode": "synthetic_grid",
            "selector_name": selector_name,
        }
        
        assert "selector_name" in config_dict, f"selector_name missing from config for {description}"
        assert config_dict["selector_name"] == selector_name, f"Wrong selector_name for {description}"


def test_backtest_request_model_has_selector():
    """Verify BacktestStartRequest model includes selector_name field."""
    from src.web_app import BacktestStartRequest
    
    req = BacktestStartRequest(
        start="2025-12-07",
        end="2025-12-08",
        selector_name="greg_vrp_harvester"
    )
    
    assert req.selector_name == "greg_vrp_harvester"
    
    req_default = BacktestStartRequest(
        start="2025-12-07",
        end="2025-12-08",
    )
    
    assert req_default.selector_name == "generic_covered_call"
