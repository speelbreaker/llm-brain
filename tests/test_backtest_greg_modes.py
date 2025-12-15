"""
Tests for Greg Backtesting Lab modes and per-strategy breakdown.

Tests the backtest_type parameter (GENERIC vs GREG_SELECTOR) and
data_source parameter for selector scans (SYNTHETIC/HARVESTER/LIVE).
"""
import pytest
from fastapi.testclient import TestClient

from src.web_app import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestGenericBacktestMode:
    """Test generic covered call backtest mode (unchanged behavior)."""

    def test_backtest_start_generic_mode_returns_200_or_409(self, client):
        """Generic mode backtest should return 200 (started) or 409 (already running)."""
        response = client.post("/api/backtest/start", json={
            "underlying": "BTC",
            "start": "2024-01-01",
            "end": "2024-01-07",
            "backtest_type": "generic",
        })
        assert response.status_code in [200, 409]

    def test_backtest_start_generic_mode_has_expected_fields(self, client):
        """Generic mode should return expected response fields."""
        response = client.post("/api/backtest/start", json={
            "underlying": "BTC",
            "start": "2024-01-01",
            "end": "2024-01-07",
            "backtest_type": "generic",
        })
        data = response.json()
        assert "started" in data or "ok" in data or "error" in data

    def test_backtest_start_default_mode_accepted(self, client):
        """Default backtest_type should be accepted (200 or 409 if running)."""
        response = client.post("/api/backtest/start", json={
            "underlying": "BTC",
            "start": "2024-01-01",
            "end": "2024-01-07",
        })
        assert response.status_code in [200, 409]


class TestGregSelectorBacktestMode:
    """Test Greg selector backtest mode."""

    def test_backtest_start_greg_selector_mode_returns_200(self, client):
        """Greg selector mode backtest should always return 200 (no conflict with generic)."""
        response = client.post("/api/backtest/start", json={
            "underlying": "BTC",
            "start": "2024-01-01",
            "end": "2024-01-07",
            "backtest_type": "greg_selector",
            "greg_underlyings": ["BTC", "ETH"],
        })
        assert response.status_code == 200

    def test_backtest_start_greg_selector_returns_correct_type(self, client):
        """Greg selector mode should return backtest_type='greg_selector'."""
        response = client.post("/api/backtest/start", json={
            "underlying": "BTC",
            "start": "2024-01-01",
            "end": "2024-01-07",
            "backtest_type": "greg_selector",
            "greg_underlyings": ["BTC"],
        })
        assert response.status_code == 200
        data = response.json()
        assert data.get("backtest_type") == "greg_selector"
        assert data.get("started") is True
        assert data.get("completed") is True

    def test_backtest_start_greg_selector_includes_strategy_summaries(self, client):
        """Greg selector mode should return strategy_summaries with proper diagnostics."""
        response = client.post("/api/backtest/start", json={
            "underlying": "BTC",
            "start": "2024-01-01",
            "end": "2024-01-07",
            "backtest_type": "greg_selector",
            "greg_underlyings": ["BTC"],
        })
        assert response.status_code == 200
        data = response.json()
        assert "strategy_summaries" in data
        assert isinstance(data["strategy_summaries"], list)
        if len(data["strategy_summaries"]) > 0:
            summary = data["strategy_summaries"][0]
            assert "pass_count" in summary
            assert "blocked_count" in summary
            assert "no_data_count" in summary
            assert "status" in summary
            assert summary["status"] in ["PASS", "BLOCKED", "NO_DATA"]

    def test_backtest_start_greg_selector_summaries_have_strategy_fields(self, client):
        """Greg selector summaries should include strategy_code and underlying for UI display."""
        response = client.post("/api/backtest/start", json={
            "underlying": "BTC",
            "start": "2024-01-01",
            "end": "2024-01-07",
            "backtest_type": "greg_selector",
            "greg_underlyings": ["BTC", "ETH"],
        })
        assert response.status_code == 200
        data = response.json()
        assert "strategy_summaries" in data
        if len(data["strategy_summaries"]) > 0:
            summary = data["strategy_summaries"][0]
            assert "strategy_code" in summary
            assert "underlying" in summary
            assert "selection_pct" in summary
            assert isinstance(summary["strategy_code"], str)
            assert summary["underlying"] in ["BTC", "ETH"]

    def test_backtest_start_greg_selector_is_synchronous(self, client):
        """Greg selector mode should be marked as synchronous execution."""
        response = client.post("/api/backtest/start", json={
            "underlying": "BTC",
            "start": "2024-01-01",
            "end": "2024-01-07",
            "backtest_type": "greg_selector",
            "greg_underlyings": ["BTC"],
        })
        assert response.status_code == 200
        data = response.json()
        assert data.get("execution_mode") == "synchronous"

    def test_backtest_start_greg_selector_includes_summary(self, client):
        """Greg selector mode should return summary data."""
        response = client.post("/api/backtest/start", json={
            "underlying": "BTC",
            "start": "2024-01-01",
            "end": "2024-01-07",
            "backtest_type": "greg_selector",
            "greg_underlyings": ["BTC"],
        })
        assert response.status_code == 200
        data = response.json()
        assert "summary" in data
        assert "total_steps" in data
        assert "greg_underlyings" in data
        assert data["greg_underlyings"] == ["BTC"]


class TestSelectorScanDataSources:
    """Test selector frequency scan with different data sources."""

    def test_selector_scan_synthetic_returns_200(self, client):
        """Synthetic data source should return 200."""
        response = client.post("/api/backtest/selector_scan", json={
            "selector_id": "greg",
            "underlyings": ["BTC"],
            "data_source": "synthetic",
            "horizon_days": 7,
            "num_paths": 1,
        })
        assert response.status_code == 200
        data = response.json()
        assert data.get("ok") is True
        assert data.get("data_source") == "synthetic"

    def test_selector_scan_synthetic_has_summary(self, client):
        """Synthetic scan should include summary data."""
        response = client.post("/api/backtest/selector_scan", json={
            "selector_id": "greg",
            "underlyings": ["BTC"],
            "data_source": "synthetic",
            "horizon_days": 7,
            "num_paths": 1,
        })
        data = response.json()
        assert "summary" in data
        assert "total_steps" in data

    def test_selector_scan_live_returns_200(self, client):
        """Live data source should return 200."""
        response = client.post("/api/backtest/selector_scan", json={
            "selector_id": "greg",
            "underlyings": ["BTC"],
            "data_source": "live",
        })
        assert response.status_code == 200
        data = response.json()
        assert "data_source" in data
        assert data.get("data_source") == "live"

    def test_selector_scan_live_includes_diagnostics(self, client):
        """Live scan should include strategy diagnostics when successful."""
        response = client.post("/api/backtest/selector_scan", json={
            "selector_id": "greg",
            "underlyings": ["BTC"],
            "data_source": "live",
        })
        data = response.json()
        if data.get("ok"):
            assert "summary" in data

    def test_selector_scan_harvester_returns_200(self, client):
        """Harvester data source should return 200."""
        response = client.post("/api/backtest/selector_scan", json={
            "selector_id": "greg",
            "underlyings": ["BTC"],
            "data_source": "harvester",
        })
        assert response.status_code == 200
        data = response.json()
        assert "data_source" in data
        assert data.get("data_source") == "harvester"

    def test_selector_scan_harvester_handles_missing_data(self, client):
        """Harvester should handle missing data gracefully."""
        response = client.post("/api/backtest/selector_scan", json={
            "selector_id": "greg",
            "underlyings": ["INVALID_UNDERLYING"],
            "data_source": "harvester",
        })
        data = response.json()
        assert "data_source" in data

    def test_selector_scan_default_is_synthetic(self, client):
        """Default data_source should be 'synthetic'."""
        response = client.post("/api/backtest/selector_scan", json={
            "selector_id": "greg",
            "underlyings": ["BTC"],
            "horizon_days": 7,
            "num_paths": 1,
        })
        data = response.json()
        assert data.get("data_source") == "synthetic" or data.get("ok") is True


class TestBacktestTypeEnum:
    """Test BacktestType enum values."""

    def test_generic_value(self):
        """BacktestType.GENERIC should have correct value."""
        from src.web.routes_backtest import BacktestType
        assert BacktestType.GENERIC.value == "generic"

    def test_greg_selector_value(self):
        """BacktestType.GREG_SELECTOR should have correct value."""
        from src.web.routes_backtest import BacktestType
        assert BacktestType.GREG_SELECTOR.value == "greg_selector"


class TestSelectorDataSourceEnum:
    """Test SelectorDataSource enum values."""

    def test_synthetic_value(self):
        """SelectorDataSource.SYNTHETIC should have correct value."""
        from src.web.routes_backtest import SelectorDataSource
        assert SelectorDataSource.SYNTHETIC.value == "synthetic"

    def test_harvester_value(self):
        """SelectorDataSource.HARVESTER should have correct value."""
        from src.web.routes_backtest import SelectorDataSource
        assert SelectorDataSource.HARVESTER.value == "harvester"

    def test_live_value(self):
        """SelectorDataSource.LIVE should have correct value."""
        from src.web.routes_backtest import SelectorDataSource
        assert SelectorDataSource.LIVE.value == "live"


class TestStrategyBacktestSummary:
    """Test StrategyBacktestSummary model."""

    def test_model_has_required_fields(self):
        """StrategyBacktestSummary should have all required fields."""
        from src.web.routes_backtest import StrategyBacktestSummary
        
        summary = StrategyBacktestSummary(
            strategy_code="A",
            strategy_name="Strategy A: ATM Straddle",
            underlying="BTC",
        )
        assert summary.bot_id == "GregBot"
        assert summary.strategy_code == "A"
        assert summary.strategy_name == "Strategy A: ATM Straddle"
        assert summary.underlying == "BTC"
        assert summary.selections == 0
        assert summary.selection_pct == 0.0
