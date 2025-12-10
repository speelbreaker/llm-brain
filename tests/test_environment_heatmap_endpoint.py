"""
Tests for the environment heatmap API endpoint.
"""
import pytest
from fastapi.testclient import TestClient

from src.web_app import app


client = TestClient(app)


def test_environment_heatmap_basic():
    """Test that the environment heatmap endpoint returns valid data."""
    payload = {
        "underlying": "BTC",
        "x_metric": "vrp_30d",
        "y_metric": "adx_14d",
        "horizon_days": 30,
        "decision_interval_days": 1,
        "x_start": 0,
        "x_step": 5,
        "x_points": 3,
        "y_start": 15,
        "y_step": 5,
        "y_points": 3,
    }
    
    response = client.post("/api/environment_heatmap", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["ok"] is True
    assert data["underlying"] == "BTC"
    assert data["x_metric"] == "vrp_30d"
    assert data["y_metric"] == "adx_14d"
    assert "x_labels" in data
    assert "y_labels" in data
    assert "grid" in data
    
    assert len(data["x_labels"]) == 3
    assert len(data["y_labels"]) == 3
    assert len(data["grid"]) == 3
    assert len(data["grid"][0]) == 3


def test_environment_heatmap_eth():
    """Test the heatmap with ETH underlying."""
    payload = {
        "underlying": "ETH",
        "x_metric": "rsi_14d",
        "y_metric": "iv_rank_6m",
        "horizon_days": 60,
        "decision_interval_days": 1,
        "x_start": 30,
        "x_step": 10,
        "x_points": 4,
        "y_start": 0.2,
        "y_step": 0.2,
        "y_points": 4,
    }
    
    response = client.post("/api/environment_heatmap", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["ok"] is True
    assert data["underlying"] == "ETH"
    assert len(data["x_labels"]) == 4
    assert len(data["y_labels"]) == 4
    assert len(data["grid"]) == 4
    for row in data["grid"]:
        assert len(row) == 4
        for cell in row:
            assert isinstance(cell, (int, float))
            assert 0 <= cell <= 100


def test_environment_heatmap_grid_values_sum():
    """Test that grid values represent valid percentages."""
    payload = {
        "underlying": "BTC",
        "x_metric": "vrp_30d",
        "y_metric": "adx_14d",
        "horizon_days": 100,
        "decision_interval_days": 1,
        "x_start": -20,
        "x_step": 20,
        "x_points": 5,
        "y_start": 10,
        "y_step": 10,
        "y_points": 5,
    }
    
    response = client.post("/api/environment_heatmap", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["ok"] is True
    
    total = sum(sum(row) for row in data["grid"])
    assert total <= 100.1
