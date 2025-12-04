"""
Web server wrapper for the Options Trading Agent.
Provides health check endpoint for Autoscale deployment while running the agent in background.
"""
import os
import threading
import time
from flask import Flask, jsonify

app = Flask(__name__)

agent_status = {
    "running": False,
    "iterations": 0,
    "last_iteration": None,
    "error": None
}


def run_agent_loop():
    """Run the agent loop in a background thread."""
    global agent_status
    
    try:
        from src.config import settings
        from src.deribit_client import DeribitClient
        from src.state_builder import build_agent_state
        from src.risk_engine import check_action_allowed
        from src.policy_rule_based import decide_action
        from src.agent_brain_llm import choose_action_with_llm
        from src.execution import execute_action
        from src.logging_utils import log_decision
        from datetime import datetime
        
        agent_status["running"] = True
        client = DeribitClient()
        
        while True:
            try:
                iteration_time = datetime.now().isoformat()
                agent_status["last_iteration"] = iteration_time
                agent_status["iterations"] += 1
                
                state = build_agent_state(client)
                candidates = state.candidate_options
                
                if settings.llm_enabled:
                    proposed = choose_action_with_llm(state, candidates)
                else:
                    proposed = decide_action(state)
                
                risk_allowed, risk_reasons = check_action_allowed(state, proposed)
                
                if risk_allowed:
                    final_action = proposed
                else:
                    final_action = {
                        "action": "DO_NOTHING",
                        "params": {},
                        "reasoning": f"Blocked by risk engine: {risk_reasons}"
                    }
                
                exec_result = execute_action(client, final_action)
                
                log_decision(
                    agent_state=state,
                    proposed_action=proposed,
                    final_action=final_action,
                    risk_allowed=risk_allowed,
                    risk_reasons=risk_reasons,
                    execution_result=exec_result,
                )
                
                time.sleep(settings.loop_interval_sec)
                
            except Exception as e:
                agent_status["error"] = str(e)
                time.sleep(60)
                
    except Exception as e:
        agent_status["running"] = False
        agent_status["error"] = str(e)


@app.route("/")
def health():
    """Health check endpoint for deployment."""
    return jsonify({
        "status": "healthy",
        "service": "options-trading-agent",
        "agent": agent_status
    })


@app.route("/health")
def health_check():
    """Explicit health check endpoint."""
    return jsonify({"status": "ok"})


def start_agent_thread():
    """Start the agent loop in a background thread."""
    global agent_status
    if not agent_status["running"]:
        print("=" * 60)
        print("IMPORTANT: This is a RESEARCH/EXPERIMENTATION system")
        print("Running on Deribit TESTNET only")
        print("This is NOT financial advice")
        print("=" * 60)
        
        agent_thread = threading.Thread(target=run_agent_loop, daemon=True)
        agent_thread.start()
        print("Agent loop started in background thread")


start_agent_thread()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting web server on port {port}")
    app.run(host="0.0.0.0", port=port)
