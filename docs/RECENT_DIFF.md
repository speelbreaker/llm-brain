# RECENT_DIFF

- generated_at_utc: 2026-01-29T19:07:38Z
- branch: fix/dashboard-relative-api-links
- head: 63b3bbe7ddbad600c0c3dd7c82cc0d826d1bed72
- base: origin/main

## Last 25 commits
63b3bbe fix(dashboard): make API calls relative to support /platform prefix
7cc3fae fix: use relative path for intraday data fetch
78dc35d fix: use relative path for API links in dashboard
1ec3d97 supervisor: keep active pointer for vault validation
30a7297 cleanup: remove obsidian vault from app
0e0d2bd supervisor: commit vault updates from app repo
4166979 supervisor: guard vault workspace boundaries
ad02572 vault: add PROMPT_004 for Phase 3 safety tests
d80b148 fix: import LLMFailure from debate, not models
321c424 queue: restore after conflict (#24)
1671630 feat: Supervisor Loop Hardening + Deterministic Fixers (#20)
ae0719a queue: done P0-002 (finalize) (#23)
59ca267 P0-002: Workflow Infrastructure (#22)
9a2554f feat(ops): add supervisor loop script for VPS orchestration (#21)
234c325 feat(workflow): implement Phase 2 vault structure + validator
b5d0bdf Ops health contract: fail-closed + generator error details (#17)
032b133 Merge pull request #18 from speelbreaker/loop-lint-target
35c0be7 fix(supervisor): target lint-only autofix to PR changed files
80d8c9c Update dashboard.py
2d3b94a changes
52afc2b Regenerate context-pack (ops health/manifest/diff)
247e65f Update context-pack artifacts
983ab41 Auto-commit 2025-12-22T20:57:23Z
6742b28 Merge pull request #14 from speelbreaker/loop-diag-hardening
6c3af3e feat(supervisor): add diag + codex capability gating (lint-only safe)

## Diff stat (origin/main..HEAD)
 src/web/dashboard.py | 144 +++++++++++++++++++++++++--------------------------
 1 file changed, 72 insertions(+), 72 deletions(-)

## Patch (origin/main..HEAD)
diff --git a/src/web/dashboard.py b/src/web/dashboard.py
index 6177474..edb0643 100644
--- a/src/web/dashboard.py
+++ b/src/web/dashboard.py
@@ -3023,7 +3023,7 @@ def render_dashboard_html() -> str:
       if (!el) return;
 
       try {{
-        const res = await fetch('/api/meta/version');
+        const res = await fetch('api/meta/version');
         const data = await res.json();
         if (!data || data.ok !== true) {{
           el.textContent = 'unknown';
@@ -3090,7 +3090,7 @@ def render_dashboard_html() -> str:
     
     async function loadSupervisorJobs() {{
       try {{
-        const resp = await fetch('/api/supervisor/jobs');
+        const resp = await fetch('api/supervisor/jobs');
         const data = await resp.json();
         
         if (data.error === 'not_configured') {{
@@ -3408,7 +3408,7 @@ def render_dashboard_html() -> str:
       expertContainer.innerHTML = '<p style="color: #666; font-style: italic;">Loading...</p>';
       
       try {{
-        const res = await fetch('/api/bots/strategies');
+        const res = await fetch('api/bots/strategies');
         const data = await res.json();
         
         if (data.ok) {{
@@ -3867,7 +3867,7 @@ def render_dashboard_html() -> str:
       }});
       
       try {{
-        const res = await fetch('/api/bots/global_risk', {{
+        const res = await fetch('api/bots/global_risk', {{
           method: 'POST',
           headers: {{'Content-Type': 'application/json'}},
           body: JSON.stringify({{
@@ -3894,7 +3894,7 @@ def render_dashboard_html() -> str:
       if (status) status.textContent = 'Resetting...';
       
       try {{
-        const res = await fetch('/api/bots/global_risk', {{
+        const res = await fetch('api/bots/global_risk', {{
           method: 'POST',
           headers: {{'Content-Type': 'application/json'}},
           body: JSON.stringify({{
@@ -4065,7 +4065,7 @@ def render_dashboard_html() -> str:
       valuesEl.innerHTML = '';
       
       try {{
-        const res = await fetch('/api/greg/calibration');
+        const res = await fetch('api/greg/calibration');
         const data = await res.json();
         
         if (data.ok) {{
@@ -4117,7 +4117,7 @@ def render_dashboard_html() -> str:
     
     async function loadGregTradingMode() {{
       try {{
-        const res = await fetch('/api/greg/trading_mode');
+        const res = await fetch('api/greg/trading_mode');
         const data = await res.json();
         if (data.ok) {{
           gregTradingMode = data.mode;
@@ -4204,7 +4204,7 @@ def render_dashboard_html() -> str:
       }});
       
       try {{
-        const res = await fetch('/api/greg/trading_mode', {{
+        const res = await fetch('api/greg/trading_mode', {{
           method: 'POST',
           headers: {{ 'Content-Type': 'application/json' }},
           body: JSON.stringify({{
@@ -4245,7 +4245,7 @@ def render_dashboard_html() -> str:
       await loadGregTradingMode();
       
       try {{
-        const res = await fetch('/api/bots/greg/management');
+        const res = await fetch('api/bots/greg/management');
         const data = await res.json();
         
         if (data.ok) {{
@@ -4273,7 +4273,7 @@ def render_dashboard_html() -> str:
       await loadGregTradingMode();
       
       try {{
-        const res = await fetch('/api/bots/greg/management/mock', {{ method: 'POST' }});
+        const res = await fetch('api/bots/greg/management/mock', {{ method: 'POST' }});
         const data = await res.json();
         
         if (data.ok) {{
@@ -4404,7 +4404,7 @@ def render_dashboard_html() -> str:
       }}
       
       try {{
-        const res = await fetch('/api/bots/greg/execute_suggestion', {{
+        const res = await fetch('api/bots/greg/execute_suggestion', {{
           method: 'POST',
           headers: {{ 'Content-Type': 'application/json' }},
           body: JSON.stringify({{
@@ -4446,7 +4446,7 @@ def render_dashboard_html() -> str:
       const dryRunBadge = document.getElementById('hedge-dry-run-badge');
       
       try {{
-        const res = await fetch('/api/bots/greg/hedging');
+        const res = await fetch('api/bots/greg/hedging');
         const data = await res.json();
         
         if (data.ok) {{
@@ -4529,7 +4529,7 @@ def render_dashboard_html() -> str:
       statusEl.textContent = 'Evaluating hedge needs...';
       
       try {{
-        const res = await fetch('/api/bots/greg/hedging/evaluate', {{ method: 'POST' }});
+        const res = await fetch('api/bots/greg/hedging/evaluate', {{ method: 'POST' }});
         const data = await res.json();
         
         if (data.ok) {{
@@ -4758,7 +4758,7 @@ def render_dashboard_html() -> str:
     async function loadSystemHealthStatus() {{
       // Load agent health guard status
       try {{
-        const healthStatusRes = await fetch('/api/system_health/status');
+        const healthStatusRes = await fetch('api/system_health/status');
         const healthStatus = await healthStatusRes.json();
         
         const healthGuardEl = document.getElementById('health-guard-status');
@@ -4794,8 +4794,8 @@ def render_dashboard_html() -> str:
       // Load LLM status and check LLM readiness
       try {{
         const [llmRes, readinessRes] = await Promise.all([
-          fetch('/api/llm_status'),
-          fetch('/api/llm_readiness')
+          fetch('api/llm_status'),
+          fetch('api/llm_readiness')
         ]);
         const llmData = await llmRes.json();
         const readinessData = await readinessRes.json();
@@ -4827,7 +4827,7 @@ def render_dashboard_html() -> str:
       
       // Load risk limits
       try {{
-        const riskRes = await fetch('/api/risk_limits');
+        const riskRes = await fetch('api/risk_limits');
         const riskData = await riskRes.json();
         if (riskData.ok) {{
           const ks = riskData.kill_switch_enabled ? 'ON' : 'OFF';
@@ -4964,7 +4964,7 @@ def render_dashboard_html() -> str:
       detailsEl.innerHTML = '';
 
       try {{
-        const res = await fetch('/api/ops/health/status');
+        const res = await fetch('api/ops/health/status');
         if (res.status === 404) {{
           statusEl.textContent = 'Status: no cached health yet';
           summaryEl.innerHTML = 'No cached health yet.';
@@ -4984,7 +4984,7 @@ def render_dashboard_html() -> str:
       const summaryEl = document.getElementById('ops-health-summary');
       if (summaryEl) summaryEl.innerHTML = '<span style="color: #666;">Running ops healthcheck...</span>';
       try {{
-        const res = await fetch('/api/ops/health/run', {{ method: 'POST' }});
+        const res = await fetch('api/ops/health/run', {{ method: 'POST' }});
         if (!res.ok) {{
           const err = await res.json();
           if (summaryEl) summaryEl.innerHTML = `<span style="color: #c62828;">${{err.error || 'Healthcheck failed'}}</span>`;
@@ -4999,7 +4999,7 @@ def render_dashboard_html() -> str:
     
     async function loadRuntimeConfig() {{
       try {{
-        const res = await fetch('/api/system/runtime-config');
+        const res = await fetch('api/system/runtime-config');
         const data = await res.json();
         if (data.ok) {{
           // Kill switch toggle
@@ -5056,7 +5056,7 @@ def render_dashboard_html() -> str:
       const labelEl = document.getElementById('kill-switch-label');
       feedbackEl.innerHTML = '<span style="color: #666;">Updating...</span>';
       try {{
-        const res = await fetch('/api/system/runtime-config', {{
+        const res = await fetch('api/system/runtime-config', {{
           method: 'POST',
           headers: {{'Content-Type': 'application/json'}},
           body: JSON.stringify({{kill_switch_enabled: enabled}})
@@ -5094,7 +5094,7 @@ def render_dashboard_html() -> str:
       const feedbackEl = document.getElementById('trade-mode-feedback');
       feedbackEl.innerHTML = '<span style="color: #666;">Updating...</span>';
       try {{
-        const res = await fetch('/api/system/runtime-config', {{
+        const res = await fetch('api/system/runtime-config', {{
           method: 'POST',
           headers: {{'Content-Type': 'application/json'}},
           body: JSON.stringify({{trade_mode: mode}})
@@ -5121,7 +5121,7 @@ def render_dashboard_html() -> str:
       const value = parseFloat(inputEl.value) || 0;
       feedbackEl.innerHTML = '<span style="color: #666;">Updating...</span>';
       try {{
-        const res = await fetch('/api/system/runtime-config', {{
+        const res = await fetch('api/system/runtime-config', {{
           method: 'POST',
           headers: {{'Content-Type': 'application/json'}},
           body: JSON.stringify({{daily_drawdown_limit_pct: value}})
@@ -5144,7 +5144,7 @@ def render_dashboard_html() -> str:
       feedbackEl.innerHTML = '<span style="color: #666;">Updating...</span>';
       const modeLabels = {{'rule_only': 'Rule Only', 'llm_only': 'LLM Only', 'hybrid_shadow': 'Hybrid (LLM Shadow)'}};
       try {{
-        const res = await fetch('/api/system/runtime-config', {{
+        const res = await fetch('api/system/runtime-config', {{
           method: 'POST',
           headers: {{'Content-Type': 'application/json'}},
           body: JSON.stringify({{decision_mode: mode}})
@@ -5167,7 +5167,7 @@ def render_dashboard_html() -> str:
       const labelEl = document.getElementById('dry-run-label');
       feedbackEl.innerHTML = '<span style="color: #666;">Updating...</span>';
       try {{
-        const res = await fetch('/api/system/runtime-config', {{
+        const res = await fetch('api/system/runtime-config', {{
           method: 'POST',
           headers: {{'Content-Type': 'application/json'}},
           body: JSON.stringify({{dry_run: enabled}})
@@ -5194,7 +5194,7 @@ def render_dashboard_html() -> str:
       feedbackEl.innerHTML = '<span style="color: #666;">Updating...</span>';
       const actionLabels = {{'halt': 'Halt', 'auto_heal': 'Auto-Heal'}};
       try {{
-        const res = await fetch('/api/system/runtime-config', {{
+        const res = await fetch('api/system/runtime-config', {{
           method: 'POST',
           headers: {{'Content-Type': 'application/json'}},
           body: JSON.stringify({{position_reconcile_action: action}})
@@ -5218,7 +5218,7 @@ def render_dashboard_html() -> str:
     async function loadLLMStrategyConfig() {{
       // Load LLM status
       try {{
-        const llmRes = await fetch('/api/llm_status');
+        const llmRes = await fetch('api/llm_status');
         const llmData = await llmRes.json();
         if (llmData.ok) {{
           document.getElementById('llm-mode-label').textContent = llmData.mode + ' / ' + llmData.decision_mode;
@@ -5244,7 +5244,7 @@ def render_dashboard_html() -> str:
       
       // Load strategy thresholds
       try {{
-        const stratRes = await fetch('/api/strategy_thresholds');
+        const stratRes = await fetch('api/strategy_thresholds');
         const stratData = await stratRes.json();
         if (stratData.ok) {{
           const eff = stratData.effective;
@@ -5265,7 +5265,7 @@ def render_dashboard_html() -> str:
       
       // Load risk limits
       try {{
-        const riskRes = await fetch('/api/risk_limits');
+        const riskRes = await fetch('api/risk_limits');
         const riskData = await riskRes.json();
         if (riskData.ok) {{
           document.getElementById('max-margin-input').value = riskData.max_margin_used_pct;
@@ -5281,7 +5281,7 @@ def render_dashboard_html() -> str:
       
       // Load reconciliation config
       try {{
-        const reconcileRes = await fetch('/api/reconciliation_config');
+        const reconcileRes = await fetch('api/reconciliation_config');
         const reconcileData = await reconcileRes.json();
         if (reconcileData.ok) {{
           document.getElementById('reconcile-action-config-select').value = reconcileData.position_reconcile_action;
@@ -5303,7 +5303,7 @@ def render_dashboard_html() -> str:
       const labelEl = document.getElementById('llm-enabled-label');
       feedbackEl.innerHTML = '<span style="color: #666;">Updating...</span>';
       try {{
-        const res = await fetch('/api/llm_status', {{
+        const res = await fetch('api/llm_status', {{
           method: 'POST',
           headers: {{'Content-Type': 'application/json'}},
           body: JSON.stringify({{llm_enabled: enabled}})
@@ -5331,7 +5331,7 @@ def render_dashboard_html() -> str:
       const exploreProb = sliderValue / 100.0;
       feedbackEl.innerHTML = '<span style="color: #666;">Updating...</span>';
       try {{
-        const res = await fetch('/api/llm_status', {{
+        const res = await fetch('api/llm_status', {{
           method: 'POST',
           headers: {{'Content-Type': 'application/json'}},
           body: JSON.stringify({{explore_prob: exploreProb}})
@@ -5352,7 +5352,7 @@ def render_dashboard_html() -> str:
       const feedbackEl = document.getElementById('training-profile-feedback');
       feedbackEl.innerHTML = '<span style="color: #666;">Updating...</span>';
       try {{
-        const res = await fetch('/api/strategy_thresholds', {{
+        const res = await fetch('api/strategy_thresholds', {{
           method: 'POST',
           headers: {{'Content-Type': 'application/json'}},
           body: JSON.stringify({{training_profile_mode: mode}})
@@ -5382,7 +5382,7 @@ def render_dashboard_html() -> str:
       }};
       
       try {{
-        const res = await fetch('/api/strategy_thresholds', {{
+        const res = await fetch('api/strategy_thresholds', {{
           method: 'POST',
           headers: {{'Content-Type': 'application/json'}},
           body: JSON.stringify(payload)
@@ -5411,7 +5411,7 @@ def render_dashboard_html() -> str:
       }};
       
       try {{
-        const res = await fetch('/api/risk_limits', {{
+        const res = await fetch('api/risk_limits', {{
           method: 'POST',
           headers: {{'Content-Type': 'application/json'}},
           body: JSON.stringify(payload)
@@ -5443,7 +5443,7 @@ def render_dashboard_html() -> str:
       }};
       
       try {{
-        const res = await fetch('/api/reconciliation_config', {{
+        const res = await fetch('api/reconciliation_config', {{
           method: 'POST',
           headers: {{'Content-Type': 'application/json'}},
           body: JSON.stringify(payload)
@@ -5464,7 +5464,7 @@ def render_dashboard_html() -> str:
       const el = document.getElementById('llm-result');
       el.innerHTML = '<span style="color: #666;">Testing LLM pipeline...</span>';
       try {{
-        const res = await fetch('/api/test_llm_decision', {{ method: 'POST' }});
+        const res = await fetch('api/test_llm_decision', {{ method: 'POST' }});
         const data = await res.json();
         if (data.ok) {{
           el.innerHTML = `<span style="color: #2e7d32;">✅ LLM OK: ${{data.action}}</span><br><span style="color: #666; font-size: 0.8em;">${{data.reasoning || ''}}</span>`;
@@ -5481,7 +5481,7 @@ def render_dashboard_html() -> str:
       const statusEl = document.getElementById('reconcile-status-line');
       el.innerHTML = '<span style="color: #666;">Running reconciliation...</span>';
       try {{
-        const res = await fetch('/api/reconcile_positions', {{ method: 'POST' }});
+        const res = await fetch('api/reconcile_positions', {{ method: 'POST' }});
         const data = await res.json();
         if (data.ok) {{
           const s = data.summary;
@@ -5505,7 +5505,7 @@ def render_dashboard_html() -> str:
       const el = document.getElementById('risk-result');
       el.innerHTML = '<span style="color: #666;">Testing risk engine...</span>';
       try {{
-        const res = await fetch('/api/test_kill_switch', {{ method: 'POST' }});
+        const res = await fetch('api/test_kill_switch', {{ method: 'POST' }});
         const data = await res.json();
         if (data.ok) {{
           if (data.allowed) {{
@@ -5536,7 +5536,7 @@ def render_dashboard_html() -> str:
       if (badge) {{ badge.textContent = 'CHECKING...'; badge.style.background = '#e0e0e0'; badge.style.color = '#666'; }}
       
       try {{
-        const res = await fetch('/api/agent_healthcheck', {{ method: 'POST' }});
+        const res = await fetch('api/agent_healthcheck', {{ method: 'POST' }});
         const data = await res.json();
         const now = new Date().toLocaleTimeString();
         
@@ -5624,7 +5624,7 @@ def render_dashboard_html() -> str:
       statusEl.style.color = '#666';
       
       try {{
-        const res = await fetch('/api/steward/run', {{ method: 'POST' }});
+        const res = await fetch('api/steward/run', {{ method: 'POST' }});
         const data = await res.json();
         renderStewardReport(data);
       }} catch (err) {{
@@ -5635,7 +5635,7 @@ def render_dashboard_html() -> str:
     
     async function loadStewardReport() {{
       try {{
-        const res = await fetch('/api/steward/report');
+        const res = await fetch('api/steward/report');
         const data = await res.json();
         renderStewardReport(data);
       }} catch (err) {{
@@ -5658,7 +5658,7 @@ def render_dashboard_html() -> str:
 
     async function fetchStatus() {{
       try {{
-        const res = await fetch('/status');
+        const res = await fetch('status');
         const data = await res.json();
         
         const spot = data.state?.spot || {{}};
@@ -5692,7 +5692,7 @@ def render_dashboard_html() -> str:
     
     async function updateStrategyStatus() {{
       try {{
-        const res = await fetch('/api/strategy-status');
+        const res = await fetch('api/strategy-status');
         const s = await res.json();
         
         const modeLabel = s.training_mode ? `Training (${{s.mode}})` : s.mode.charAt(0).toUpperCase() + s.mode.slice(1);
@@ -5747,7 +5747,7 @@ def render_dashboard_html() -> str:
     
     async function updateClosedPositions() {{
       try {{
-        const res = await fetch('/api/positions/closed');
+        const res = await fetch('api/positions/closed');
         const data = await res.json();
         const tbody = document.getElementById('live-closed-positions-body');
         const chains = data.chains || [];
@@ -5829,7 +5829,7 @@ def render_dashboard_html() -> str:
       const summaryEl = document.getElementById('positions-pnl-summary');
       
       try {{
-        const res = await fetch('/api/positions/open');
+        const res = await fetch('api/positions/open');
         const data = await res.json();
         const positions = data.positions || [];
         const totals = data.totals || {{}};
@@ -5895,7 +5895,7 @@ def render_dashboard_html() -> str:
     
     async function fetchDecisions() {{
       try {{
-        const res = await fetch('/api/agent/decisions');
+        const res = await fetch('api/agent/decisions');
         const data = await res.json();
         
         const decisions = data.decisions || [];
@@ -6039,7 +6039,7 @@ def render_dashboard_html() -> str:
     
     async function fetchBacktestRuns() {{
       try {{
-        const res = await fetch('/api/backtests');
+        const res = await fetch('api/backtests');
         const runs = await res.json();
         renderBacktestRuns(runs);
       }} catch (err) {{
@@ -6088,7 +6088,7 @@ def render_dashboard_html() -> str:
             <td>${{numTrades}}</td>
             <td>
               <button onclick="viewRunDetail('${{run.run_id}}')" style="background:#2196f3;color:#fff;border:none;padding:4px 8px;border-radius:3px;cursor:pointer;margin-right:4px;font-size:0.8em;">View</button>
-              <a href="/api/backtests/${{run.run_id}}/download" style="background:#4caf50;color:#fff;border:none;padding:4px 8px;border-radius:3px;cursor:pointer;text-decoration:none;font-size:0.8em;">Download</a>
+              <a href="api/backtests/${{run.run_id}}/download" style="background:#4caf50;color:#fff;border:none;padding:4px 8px;border-radius:3px;cursor:pointer;text-decoration:none;font-size:0.8em;">Download</a>
             </td>
           </tr>
         `;
@@ -6097,7 +6097,7 @@ def render_dashboard_html() -> str:
     
     async function viewRunDetail(runId) {{
       try {{
-        const res = await fetch('/api/backtests/' + runId);
+        const res = await fetch('api/backtests/' + runId);
         if (!res.ok) {{
           alert('Failed to load run details');
           return;
@@ -6171,7 +6171,7 @@ def render_dashboard_html() -> str:
     
     async function loadChatHistory() {{
       try {{
-        const res = await fetch('/chat/messages');
+        const res = await fetch('chat/messages');
         const data = await res.json();
         renderChatMessages(data.messages || []);
       }} catch (err) {{
@@ -6199,7 +6199,7 @@ def render_dashboard_html() -> str:
       document.getElementById('chat-messages').scrollTop = document.getElementById('chat-messages').scrollHeight;
 
       try {{
-        const res = await fetch('/chat', {{
+        const res = await fetch('chat', {{
           method: 'POST',
           headers: {{ 'Content-Type': 'application/json' }},
           body: JSON.stringify({{ question: q }})
@@ -6224,7 +6224,7 @@ def render_dashboard_html() -> str:
     
     async function clearChat() {{
       try {{
-        await fetch('/chat/clear', {{ method: 'POST' }});
+        await fetch('chat/clear', {{ method: 'POST' }});
         renderChatMessages([]);
       }} catch (err) {{
         console.error('Failed to clear chat:', err);
@@ -6662,7 +6662,7 @@ def render_dashboard_html() -> str:
       container.innerHTML = '<div style="text-align:center;color:#666;grid-column:1/-1;">Loading...</div>';
       
       try {{
-        const res = await fetch('/api/calibration/auto_status');
+        const res = await fetch('api/calibration/auto_status');
         if (!res.ok) throw new Error(`HTTP ${{res.status}}`);
         const data = await res.json();
         
@@ -6734,7 +6734,7 @@ def render_dashboard_html() -> str:
         btn.innerText = 'Applying...';
         
         try {{
-          const res = await fetch('/api/calibration/apply_direct', {{
+          const res = await fetch('api/calibration/apply_direct', {{
             method: 'POST',
             headers: {{ 'Content-Type': 'application/json' }},
             body: JSON.stringify(lastCalibrationResult)
@@ -6770,7 +6770,7 @@ def render_dashboard_html() -> str:
       btn.innerText = 'Applying...';
       
       try {{
-        const res = await fetch('/api/calibration/use_latest', {{
+        const res = await fetch('api/calibration/use_latest', {{
           method: 'POST',
           headers: {{ 'Content-Type': 'application/json' }},
           body: JSON.stringify({{ underlying, dte_min: dteMin, dte_max: dteMax }})
@@ -6929,7 +6929,7 @@ def render_dashboard_html() -> str:
       _setFidelityDashboardLoading();
 
       try {{
-        const res = await fetch('/api/fidelity/latest');
+        const res = await fetch('api/fidelity/latest');
         const data = res.ok ? await res.json() : null;
         _applyFidelityDashboardReport(data);
 
@@ -6957,7 +6957,7 @@ def render_dashboard_html() -> str:
     
     async function fetchPolicy() {{
       try {{
-        const res = await fetch('/api/calibration/policy');
+        const res = await fetch('api/calibration/policy');
         if (!res.ok) throw new Error(`HTTP ${{res.status}}`);
         const data = await res.json();
         
@@ -7130,7 +7130,7 @@ def render_dashboard_html() -> str:
     
     async function toggleTraining(enable) {{
       try {{
-        const res = await fetch('/api/training/toggle', {{
+        const res = await fetch('api/training/toggle', {{
           method: 'POST',
           headers: {{ 'Content-Type': 'application/json' }},
           body: JSON.stringify({{ enable }})
@@ -7270,7 +7270,7 @@ def render_dashboard_html() -> str:
       document.getElementById('bt-error').style.display = 'none';
       
       try {{
-        const res = await fetch('/api/backtest/start', {{
+        const res = await fetch('api/backtest/start', {{
           method: 'POST',
           headers: {{ 'Content-Type': 'application/json' }},
           body: JSON.stringify(payload),
@@ -7404,9 +7404,9 @@ def render_dashboard_html() -> str:
       
       try {{
         if (isPaused) {{
-          await fetch('/api/backtest/resume', {{ method: 'POST' }});
+          await fetch('api/backtest/resume', {{ method: 'POST' }});
         }} else {{
-          await fetch('/api/backtest/pause', {{ method: 'POST' }});
+          await fetch('api/backtest/pause', {{ method: 'POST' }});
         }}
       }} catch (err) {{
         console.error('Pause/Resume error:', err);
@@ -7415,7 +7415,7 @@ def render_dashboard_html() -> str:
     
     async function stopBacktest() {{
       try {{
-        await fetch('/api/backtest/stop', {{ method: 'POST' }});
+        await fetch('api/backtest/stop', {{ method: 'POST' }});
       }} catch (err) {{
         console.error('Stop backtest error:', err);
       }}
@@ -7423,7 +7423,7 @@ def render_dashboard_html() -> str:
     
     async function refreshBacktestStatus() {{
       try {{
-        const res = await fetch('/api/backtest/status');
+        const res = await fetch('api/backtest/status');
         if (!res.ok) return;
         const st = await res.json();
         
@@ -7647,7 +7647,7 @@ def render_dashboard_html() -> str:
       document.getElementById('insights-box').style.display = 'none';
       
       try {{
-        const res = await fetch('/api/backtest/run', {{
+        const res = await fetch('api/backtest/run', {{
           method: 'POST',
           headers: {{ 'Content-Type': 'application/json' }},
           body: JSON.stringify({{
@@ -7852,7 +7852,7 @@ def render_dashboard_html() -> str:
       box.innerText = 'Generating insights...';
       
       try {{
-        const res = await fetch('/api/backtest/insights', {{
+        const res = await fetch('api/backtest/insights', {{
           method: 'POST',
           headers: {{ 'Content-Type': 'application/json' }},
           body: JSON.stringify({{
@@ -7909,7 +7909,7 @@ def render_dashboard_html() -> str:
       }};
       
       try {{
-        const res = await fetch('/api/backtest/selector_scan', {{
+        const res = await fetch('api/backtest/selector_scan', {{
           method: 'POST',
           headers: {{ 'Content-Type': 'application/json' }},
           body: JSON.stringify(payload)
@@ -7952,7 +7952,7 @@ def render_dashboard_html() -> str:
       }}
 
       try {{
-        const resp = await fetch("/api/data_status/intraday");
+        const resp = await fetch("api/data_status/intraday");
         const data = await resp.json();
 
         if (!data.ok) {{
@@ -8051,7 +8051,7 @@ def render_dashboard_html() -> str:
       }};
       
       try {{
-        const res = await fetch('/api/backtest/selector_heatmap', {{
+        const res = await fetch('api/backtest/selector_heatmap', {{
           method: 'POST',
           headers: {{ 'Content-Type': 'application/json' }},
           body: JSON.stringify(payload)
@@ -8129,7 +8129,7 @@ def render_dashboard_html() -> str:
       }};
       
       try {{
-        const res = await fetch('/api/environment_heatmap', {{
+        const res = await fetch('api/environment_heatmap', {{
           method: 'POST',
           headers: {{ 'Content-Type': 'application/json' }},
           body: JSON.stringify(payload)
@@ -8231,7 +8231,7 @@ def render_dashboard_html() -> str:
         sweetStatus.textContent = 'Loading sweet spots...';
         sweetStatus.style.color = '';
         
-        fetch('/api/greg_sweetspots')
+        fetch('api/greg_sweetspots')
           .then(r => r.json())
           .then(renderSweetSpots)
           .catch(err => {{
@@ -8247,7 +8247,7 @@ def render_dashboard_html() -> str:
         if (runBtn) runBtn.disabled = true;
         if (refreshBtn) refreshBtn.disabled = true;
         
-        fetch('/api/greg_sweetspots/run', {{ method: 'POST' }})
+        fetch('api/greg_sweetspots/run', {{ method: 'POST' }})
           .then(r => r.json())
           .then(payload => {{
             if (!payload.ok) {{
