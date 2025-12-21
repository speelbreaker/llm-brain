from unittest.mock import patch
from fastapi.testclient import TestClient
from src.supervisor.app import app
from src.supervisor.models_mvp import DecisionTrace
from src.supervisor.config import SupervisorSettings

def test_list_traces(tmp_path):
    # Create dummy traces
    trace_file = tmp_path / "mvp_traces.jsonl"
    t1 = DecisionTrace(bot_id="bot1", strategy_id="s1", mode="paper", run_reason="manual", underlying="BTC", spot=100.0, narrative="n1", decision="NO_TRADE")
    t2 = DecisionTrace(
        bot_id="bot1", strategy_id="s1", mode="paper", run_reason="manual", underlying="BTC", spot=101.0, narrative="n2", decision="TRADE", 
        chosen={"instrument_name":"foo", "expiry":"2023-01-01", "strike":100, "delta":0.5, "dte":10, "mark":10, "premium":1, "iv":50, "score":10, "otm_pct":0.1}
    )
    
    # Ensure dir exists (TraceStore usually does this, but we are writing directly first)
    trace_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(trace_file, "w") as f:
        f.write(t1.model_dump_json() + "\n")
        f.write(t2.model_dump_json() + "\n")
        
    # Mock get_settings to return settings with tmp_path as base_jobs_dir
    settings = SupervisorSettings(base_jobs_dir=str(tmp_path))
    
    with patch("src.supervisor.trace_store.get_settings", return_value=settings):
        # We also need to configure app.state.settings for the endpoint redaction
        app.state.settings = settings
        with TestClient(app) as client:
            resp = client.get("/api/mvp/traces")
            assert resp.status_code == 200
            data = resp.json()
            assert len(data) == 2
            # list_traces returns newest first (reversed)
            assert data[0]["narrative"] == "n2"
            assert data[1]["narrative"] == "n1"
            
            # Check secret hygiene (just a basic check)
            assert "sk-" not in str(data)

def test_get_trace(tmp_path):
    trace_file = tmp_path / "mvp_traces.jsonl"
    trace_file.parent.mkdir(parents=True, exist_ok=True)
    t1 = DecisionTrace(bot_id="bot1", strategy_id="s1", mode="paper", run_reason="manual", underlying="BTC", spot=100.0, narrative="n1", decision="NO_TRADE")
    
    with open(trace_file, "w") as f:
        f.write(t1.model_dump_json() + "\n")
        
    settings = SupervisorSettings(base_jobs_dir=str(tmp_path))
    
    with patch("src.supervisor.trace_store.get_settings", return_value=settings):
        app.state.settings = settings
        with TestClient(app) as client:
            resp = client.get(f"/api/mvp/traces/{t1.trace_id}")
            assert resp.status_code == 200
            data = resp.json()
            assert data["trace_id"] == t1.trace_id
            
            # Not found
            resp = client.get("/api/mvp/traces/bad-id")
            assert resp.status_code == 404
