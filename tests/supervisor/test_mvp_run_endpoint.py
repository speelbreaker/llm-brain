from unittest.mock import patch
from fastapi.testclient import TestClient
from src.supervisor.app import app
from src.supervisor.models_mvp import DecisionTrace
from src.supervisor.config import SupervisorSettings

def test_run_mvp_endpoint(tmp_path):
    settings = SupervisorSettings(base_jobs_dir=str(tmp_path))
    app.state.settings = settings
    
    # Mock run_mvp_cycle
    dummy_trace = DecisionTrace(
        bot_id="test", strategy_id="test", mode="test", run_reason="manual", 
        underlying="BTC", spot=100, narrative="test run", decision="NO_TRADE"
    )
    
    with patch("src.supervisor.app.run_mvp_cycle", return_value=dummy_trace) as mock_run:
        with patch("src.supervisor.app._is_local_request", return_value=True):
            with TestClient(app) as client:
                resp = client.post("/api/mvp/run")
                assert resp.status_code == 200
                assert resp.json()["trace_id"] == dummy_trace.trace_id
                mock_run.assert_called_once()

def test_run_mvp_forbidden_remote():
    with patch("src.supervisor.app._is_local_request", return_value=False):
        with TestClient(app) as client:
            resp = client.post("/api/mvp/run")
            assert resp.status_code == 403
