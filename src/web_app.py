"""
FastAPI web application for the Options Trading Agent.
Provides live status, chat interface, and web dashboard.
"""
from __future__ import annotations

import threading
from typing import Any, Dict

from fastapi import FastAPI, Body
from fastapi.responses import HTMLResponse, JSONResponse

from agent_loop import run_agent_loop_forever
from src.status_store import status_store
from src.chat_with_agent import chat_with_agent
from src.config import settings


app = FastAPI(
    title="Options Trading Agent Dashboard",
    description="Deribit testnet covered-call agent with live status and chat.",
    version="0.1.0",
)


def _agent_thread_target() -> None:
    """Run the agent loop forever, updating status_store each iteration."""
    def status_callback(snapshot: Dict[str, Any]) -> None:
        status_store.update(snapshot)

    run_agent_loop_forever(status_callback=status_callback)


@app.on_event("startup")
def start_background_agent() -> None:
    """Start the agent loop in a background thread on FastAPI startup."""
    print("\n" + "=" * 60)
    print("IMPORTANT: This is a RESEARCH/EXPERIMENTATION system")
    print("Running on Deribit TESTNET only")
    print("This is NOT financial advice")
    print("=" * 60 + "\n")
    
    thread = threading.Thread(target=_agent_thread_target, daemon=True)
    thread.start()
    print("Agent loop started in background thread")


@app.get("/status")
def get_status() -> JSONResponse:
    """Return the latest agent status snapshot."""
    data = status_store.get()
    return JSONResponse(content=data)


@app.get("/health")
def health_check() -> JSONResponse:
    """Health check endpoint for deployment."""
    return JSONResponse(content={"status": "healthy", "service": "options-trading-agent"})


@app.post("/chat")
def chat_endpoint(
    payload: Dict[str, Any] = Body(..., example={"question": "Why did you pick the 97k call?"}),
) -> JSONResponse:
    """
    Ask the agent a question about its recent behavior.
    Uses log files + OpenAI to generate an answer.
    """
    question = payload.get("question", "").strip()
    if not question:
        return JSONResponse(
            status_code=400,
            content={"error": "Missing 'question' field in request body."},
        )

    try:
        answer = chat_with_agent(question, log_limit=20)
        return JSONResponse(content={"question": question, "answer": answer})
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to generate answer: {str(e)}"},
        )


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    """Minimal HTML dashboard for live status + chat."""
    decision_mode = "LLM" if settings.llm_enabled else "Rule-based"
    op_mode = settings.mode.upper()
    explore_pct = int(settings.explore_prob * 100)
    
    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Options Agent Dashboard</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      max-width: 1000px;
      margin: 0 auto;
      padding: 1rem;
      background: #f8f9fa;
      color: #333;
    }}
    h1 {{
      color: #1a1a2e;
      margin-bottom: 0.25rem;
      font-size: 1.75rem;
    }}
    .subtitle {{
      color: #666;
      margin-bottom: 1.5rem;
    }}
    .badge {{
      display: inline-block;
      padding: 0.25rem 0.5rem;
      border-radius: 4px;
      font-size: 0.85rem;
      font-weight: 600;
      margin-right: 0.5rem;
    }}
    .badge-mode {{ background: #e3f2fd; color: #1565c0; }}
    .badge-research {{ background: #f3e5f5; color: #7b1fa2; }}
    .badge-production {{ background: #e8f5e9; color: #2e7d32; }}
    .badge-dry {{ background: #fff3e0; color: #e65100; }}
    .badge-live {{ background: #e8f5e9; color: #2e7d32; }}
    .section {{
      margin: 1.5rem 0;
      padding: 1.25rem;
      background: white;
      border: 1px solid #dee2e6;
      border-radius: 8px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }}
    .section h2 {{
      margin-top: 0;
      color: #1a1a2e;
      font-size: 1.25rem;
      border-bottom: 2px solid #e9ecef;
      padding-bottom: 0.5rem;
    }}
    pre {{
      background: #f1f3f4;
      padding: 1rem;
      border-radius: 6px;
      overflow: auto;
      max-height: 400px;
      font-size: 0.85rem;
      line-height: 1.4;
      white-space: pre-wrap;
      word-wrap: break-word;
    }}
    .status-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 1rem;
      margin-bottom: 1rem;
    }}
    .status-card {{
      background: #f8f9fa;
      padding: 1rem;
      border-radius: 6px;
      text-align: center;
    }}
    .status-card .label {{
      font-size: 0.85rem;
      color: #666;
      margin-bottom: 0.25rem;
    }}
    .status-card .value {{
      font-size: 1.5rem;
      font-weight: 700;
      color: #1a1a2e;
    }}
    label {{
      font-weight: 600;
      color: #333;
    }}
    textarea {{
      width: 100%;
      height: 80px;
      padding: 0.75rem;
      border: 1px solid #ced4da;
      border-radius: 6px;
      font-family: inherit;
      font-size: 0.95rem;
      resize: vertical;
      margin: 0.5rem 0;
    }}
    textarea:focus {{
      outline: none;
      border-color: #1565c0;
      box-shadow: 0 0 0 3px rgba(21, 101, 192, 0.1);
    }}
    button {{
      padding: 0.6rem 1.5rem;
      background: #1565c0;
      color: white;
      border: none;
      border-radius: 6px;
      font-size: 0.95rem;
      font-weight: 600;
      cursor: pointer;
      transition: background 0.2s;
    }}
    button:hover {{
      background: #0d47a1;
    }}
    button:disabled {{
      background: #90a4ae;
      cursor: not-allowed;
    }}
    .answer-box {{
      background: #e8f5e9;
      border-left: 4px solid #4caf50;
    }}
    .last-update {{
      font-size: 0.8rem;
      color: #888;
      text-align: right;
      margin-top: 0.5rem;
    }}
    .warning {{
      background: #fff3e0;
      border: 1px solid #ffcc80;
      padding: 0.75rem;
      border-radius: 6px;
      margin-bottom: 1rem;
      font-size: 0.9rem;
    }}
  </style>
</head>
<body>
  <h1>Options Trading Agent</h1>
  <p class="subtitle">Deribit Testnet - Covered Call Strategy</p>
  
  <div>
    <span class="badge badge-mode">{decision_mode}</span>
    <span class="badge {'badge-research' if settings.is_research else 'badge-production'}">
      {op_mode} ({explore_pct}% explore)
    </span>
    <span class="badge {'badge-dry' if settings.dry_run else 'badge-live'}">
      {'DRY RUN' if settings.dry_run else 'LIVE TRADING'}
    </span>
  </div>
  
  <div class="warning">
    This is a RESEARCH/EXPERIMENTATION system running on Deribit TESTNET only. This is NOT financial advice.
  </div>

  <div class="section">
    <h2>Live Status</h2>
    <div class="status-grid" id="status-cards">
      <div class="status-card">
        <div class="label">BTC Price</div>
        <div class="value" id="btc-price">--</div>
      </div>
      <div class="status-card">
        <div class="label">ETH Price</div>
        <div class="value" id="eth-price">--</div>
      </div>
      <div class="status-card">
        <div class="label">Portfolio</div>
        <div class="value" id="portfolio-value">--</div>
      </div>
      <div class="status-card">
        <div class="label">Last Action</div>
        <div class="value" id="last-action">--</div>
      </div>
    </div>
    <details>
      <summary style="cursor: pointer; font-weight: 600; margin-bottom: 0.5rem;">Full Status JSON</summary>
      <pre id="status-box">Loading...</pre>
    </details>
    <div class="last-update" id="last-update">Last update: --</div>
  </div>

  <div class="section">
    <h2>Chat with Agent</h2>
    <label for="question">Ask a question about the agent's behavior:</label>
    <textarea id="question" placeholder="e.g., Why did you pick the 97k call? What would you do next?"></textarea>
    <button id="ask-btn" onclick="sendQuestion()">Ask Agent</button>
    
    <h3 style="margin-top: 1.5rem;">Answer</h3>
    <pre id="answer-box" class="answer-box">Ask a question to get started...</pre>
  </div>

  <script>
    function formatNumber(num) {{
      if (num === undefined || num === null) return '--';
      if (num >= 1000000) return '$' + (num / 1000000).toFixed(2) + 'M';
      if (num >= 1000) return '$' + (num / 1000).toFixed(1) + 'K';
      return '$' + num.toFixed(0);
    }}

    async function fetchStatus() {{
      try {{
        const res = await fetch('/status');
        const data = await res.json();
        
        // Update status cards
        const spot = data.state?.spot || {{}};
        const portfolio = data.state?.portfolio || {{}};
        const finalAction = data.final_action?.action || 'Starting...';
        
        document.getElementById('btc-price').innerText = spot.BTC ? '$' + spot.BTC.toLocaleString() : '--';
        document.getElementById('eth-price').innerText = spot.ETH ? '$' + spot.ETH.toLocaleString() : '--';
        document.getElementById('portfolio-value').innerText = formatNumber(portfolio.equity_usd);
        document.getElementById('last-action').innerText = finalAction.replace('_', ' ');
        
        // Update full JSON
        document.getElementById('status-box').innerText = JSON.stringify(data, null, 2);
        document.getElementById('last-update').innerText = 'Last update: ' + new Date().toLocaleTimeString();
      }} catch (err) {{
        document.getElementById('status-box').innerText = 'Error fetching status: ' + err;
      }}
    }}

    async function sendQuestion() {{
      const q = document.getElementById('question').value;
      if (!q.trim()) {{
        alert('Please enter a question first.');
        return;
      }}
      
      const btn = document.getElementById('ask-btn');
      btn.disabled = true;
      btn.innerText = 'Thinking...';
      document.getElementById('answer-box').innerText = 'Analyzing logs and generating response...';

      try {{
        const res = await fetch('/chat', {{
          method: 'POST',
          headers: {{ 'Content-Type': 'application/json' }},
          body: JSON.stringify({{ question: q }})
        }});
        const data = await res.json();
        if (data.error) {{
          document.getElementById('answer-box').innerText = 'Error: ' + data.error;
        }} else {{
          document.getElementById('answer-box').innerText = data.answer;
        }}
      }} catch (err) {{
        document.getElementById('answer-box').innerText = 'Error sending question: ' + err;
      }} finally {{
        btn.disabled = false;
        btn.innerText = 'Ask Agent';
      }}
    }}

    // Poll status every 5 seconds
    fetchStatus();
    setInterval(fetchStatus, 5000);
    
    // Allow Enter key to submit question
    document.getElementById('question').addEventListener('keydown', function(e) {{
      if (e.key === 'Enter' && !e.shiftKey) {{
        e.preventDefault();
        sendQuestion();
      }}
    }});
  </script>
</body>
</html>
"""
