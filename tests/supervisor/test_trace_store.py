from src.supervisor.trace_store import TraceStore
from src.supervisor.models_mvp import DecisionTrace

def test_trace_store_corrupt_line(tmp_path):
    path = tmp_path / "mvp_traces.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "w") as f:
        f.write('{"valid": "json"}\n') # valid json but fails model validation
        f.write("BROKEN LINE\n")
        
        t1 = DecisionTrace(bot_id="b", strategy_id="s", mode="m", run_reason="r", underlying="u", spot=1, narrative="n", decision="d")
        f.write(t1.model_dump_json() + "\n")
        
    store = TraceStore(str(path))
    traces = store.list_traces()
    
    # Should skip broken line and invalid model
    assert len(traces) == 1
    assert traces[0].trace_id == t1.trace_id

def test_trace_store_persistence(tmp_path):
    path = tmp_path / "mvp_traces.jsonl"
    store = TraceStore(str(path))
    
    t1 = DecisionTrace(bot_id="b", strategy_id="s", mode="m", run_reason="r", underlying="u", spot=1, narrative="n", decision="d")
    store.append_trace(t1)
    
    traces = store.list_traces()
    assert len(traces) == 1
    assert traces[0].trace_id == t1.trace_id
    
    # Verify file content
    with open(path, "r") as f:
        lines = f.readlines()
        assert len(lines) == 1
        assert t1.trace_id in lines[0]
