# Recent Diff

generated_at_utc: 2025-12-22T16:39:46Z
branch: main
head_sha: 3bc7e86546adf507d277ca9f65342246a7adc804
base: origin/main

## git log --oneline -n 25
3bc7e86 Ops health per-underlying gates + facts resolver
522ef78 feat(fidelity): deterministic per-underlying latest + source/path metadata + tests
e5046b2 Merge pull request #5 from speelbreaker/security-secret-hygiene-clean
95d9b23 test: ensure bearer redaction hits regex threshold
e736d0d test: relax redaction assertions for field prefixes
7169ad1 test: align redaction minimal fixture with current logic
bec300c test: add secret tripwire checks
940dbdf chore: add gitleaks scan helpers
fe3a213 chore: enforce secret scanning gate + add leak drill (clean)
e5f7487 feat(fidelity): fix missing-close false profit, unify fidelity modules, add fidelity endpoints+tests+UI badge
07e5708 Test: MVP underlying extraction backcompat
0498e93 Fix review-28 risks: fidelity fallback + CI robustness
422bee1 Resolve stash conflict + DB sqlite support
cdf29b6 Harden fidelity MVP + restore legacy API
4413b6e Add MVP fidelity suite endpoints and UI
ac51b37 Roadmap: define calibration acceptance criteria and fidelity gate
ae29c66 Add build SHA indicator endpoint and UI
bba4c05 Add fidelity suite UI + API endpoints
ef5aa6f test
bd1d2e7 Fix harvested calibration RV scaling
84d4601 Block Deribit live_chain historically
c403e9c Backtest correctness: no overlap + linear USDC + cap expiry
e39b4a7 Finish regime IV plumbing + add provenance summary
ecd19fa Saved progress at the end of the loop
007d783 Improve backtesting by adding historical defaults and fallback logic

## git diff --stat origin/main..HEAD
 .vscode/tasks.json                                 |   2 +-
 HEALTHCHECK.md                                     |  93 ++++-
 agent_loop.py                                      |  21 +
 ...uct-minded-engineer-Update-th_1766256235012.txt |  73 ----
 data/backtests/index.jsonl                         |  18 +
 .../history/20251219_234717/fidelity_report.json   | 131 ------
 .../BTC/history/20251219_234717/fidelity_report.md | 147 -------
 data/fidelity_runs/BTC/latest/fidelity_report.json | 131 ------
 data/fidelity_runs/BTC/latest/fidelity_report.md   | 147 -------
 scripts/print_fidelity_summary.py                  |  94 +++++
 scripts/run_fidelity_from_lab.py                   |  56 +++
 scripts/run_fidelity_from_lab_daily.py             |  56 +++
 scripts/sabotage_fidelity_drill.py                 |  42 ++
 src/backtest/compare.py                            | 337 ++++++++++------
 src/backtest/covered_call_simulator.py             |   7 +
 src/backtest/diff.py                               | 169 ++++----
 src/backtest/fidelity_suite.py                     | 239 +++++++++++
 src/backtest/live_deribit_data_source.py           |  29 +-
 src/backtest/manager.py                            | 147 ++++++-
 src/backtest/pricing.py                            |  12 +-
 src/backtest/state_builder.py                      |  32 +-
 src/backtest/units.py                              |  58 +++
 src/calibration_config.py                          |   1 +
 src/calibration_extended.py                        |   3 +
 src/calibration_update_policy.py                   |  82 +++-
 src/config.py                                      |  30 ++
 src/data/live_deribit_exam.py                      |  37 ++
 src/db/__init__.py                                 |   1 +
 src/db/models_telegram.py                          |  21 +
 src/fidelity/canonical_strategies.py               | 194 ++++++++-
 src/fidelity/gating.py                             |  17 +
 src/fidelity/market_replay.py                      | 209 +++++++++-
 src/fidelity/reporting.py                          |   7 +
 src/fidelity/run_suite.py                          | 109 ++++-
 src/harvest_status.py                              | 271 +++++++++++++
 src/ops/calibration_status.py                      |  98 +++++
 src/ops/facts_resolver.py                          |  89 +++++
 src/ops/gate_factories.py                          | 442 ++++++++++++++++++++
 src/ops/gates.py                                   | 112 ++++++
 src/telegram/__init__.py                           |   1 +
 src/telegram/store.py                              | 108 +++++
 src/web/api_errors.py                              |  41 ++
 src/web/dashboard.py                               | 312 +++++++++------
 src/web/routes_backtest.py                         | 131 ++++--
 src/web/routes_fidelity.py                         |  46 +++
 src/web/routes_health.py                           |  37 +-
 src/web/routes_positions.py                        |  12 +
 src/web/routes_telegram.py                         | 214 ++++++++++
 src/web_app.py                                     |  18 +
 tests/test_api_calibration_run_with_policy.py      |  54 +++
 tests/test_api_fidelity_endpoints.py               |  64 +++
 tests/test_backtest_greg_modes.py                  |  48 ++-
 tests/test_backtest_preflight.py                   | 214 ++++++++++
 tests/test_calibration_update_policy.py            |   2 +
 tests/test_fidelity_lab_scoring.py                 |  25 ++
 tests/test_fidelity_missing_close.py               | 116 ++++++
 ..._fidelity_moneyness_fallback_and_diagnostics.py |  68 ++++
 tests/test_health_and_calibration_automation.py    |  93 ++++-
 tests/test_healthcheck_basic.py                    |  46 ++-
 tests/test_healthcheck_config.py                   |  15 +-
 tests/test_live_deribit_units.py                   |  53 +++
 tests/test_ops_health_endpoints.py                 | 258 ++++++++++++
 tests/web/expected_routes.json                     | 443 +++++++++++----------
 tests/web/test_telegram_webhook.py                 | 126 ++++++
 64 files changed, 5003 insertions(+), 1306 deletions(-)

## git diff origin/main..HEAD
diff --git a/.vscode/tasks.json b/.vscode/tasks.json
index 4b86157..31ea0c7 100644
--- a/.vscode/tasks.json
+++ b/.vscode/tasks.json
@@ -101,7 +101,7 @@
 			"command": "/bin/zsh",
 			"args": [
 				"-lc",
-				"${workspaceFolder}/.venv/bin/python -m pytest -q > /tmp/pytest_out.txt 2>&1; echo \"EXIT:$?\"; tail -n 200 /tmp/pytest_out.txt"
+				"cd \"${workspaceFolder}\" && .venv/bin/python -m pytest -q > /tmp/pytest_out.txt 2>&1; echo \"EXIT:$?\"; tail -n 200 /tmp/pytest_out.txt"
 			],
 			"isBackground": false,
 			"group": "test"
diff --git a/HEALTHCHECK.md b/HEALTHCHECK.md
index 93e6324..6ac1da7 100644
--- a/HEALTHCHECK.md
+++ b/HEALTHCHECK.md
@@ -4,7 +4,98 @@ This document lists quick commands to verify that core parts of the system are w
 
 ---
 
-## Quick Smoke Tests
+## Operational Health (watchdog-ready)
+
+This is the ops-grade health model used by automation and dashboards. It is designed for watchdogs and runtime guardrails.
+
+### Endpoints
+
+- `GET /api/ops/health/status` (cached)
+- `POST /api/ops/health/run` (force refresh + cache)
+
+### Three Layers
+
+- **Liveness**: Core pipeline checks (config validity, Deribit connectivity, state builder).
+- **Truth → Trust → Trade**:
+  - **Truth (facts)**: raw observations from the filesystem/stores (harvest presence + age, calibration last run, fidelity last run).
+  - **Trust (gates)**: normalized gate results with explicit `mode` (`off|warn|block`) and `status` (`PASS|WARN|FAIL`).
+  - **Trade (policy)**: aggregated `gate_overall` (`status|severity|can_trade`) used by dashboards and automation.
+
+### Thresholds & Policies
+
+- **Harvest freshness**:
+  - OK: `age_minutes <= 60`
+  - WARN: `60 < age_minutes <= 180`
+  - FAIL: `age_minutes > 180` or missing files
+- **Calibration freshness**:
+  - OK: `last_calibration_at <= 36h` and applied
+  - WARN: `36-72h` or `applied=False`
+  - FAIL: `>72h`, missing bundle, or last run failed
+- **Fidelity gate**:
+  - TRUSTED: OK
+  - WARNING: WARN (degraded)
+  - UNTRUSTED: WARN by default; `HEALTH_STRICT_SYNTHETIC_GATE=1` escalates to FATAL + `can_trade=False`
+  - Missing: WARN in research mode; FAIL in strict mode
+
+### Manual Ops Commands
+
+```bash
+curl -s http://localhost:5000/api/ops/health/status
+curl -s -X POST http://localhost:5000/api/ops/health/run
+```
+
+> When `OPS_HEALTH_RUN_SECRET` is defined, add `-H "X-OPS-HEALTH-SECRET: $OPS_HEALTH_RUN_SECRET"` to the guarded POST so only authorized tooling can refresh the cache.
+
+### Guarding the Ops Health endpoint
+
+- `POST /api/ops/health/run` will reject requests without the matching `X-OPS-HEALTH-SECRET` header whenever `OPS_HEALTH_RUN_SECRET` is set. The handshake ensures operators cannot accidentally hammer Deribit / synthetic-data gates from a public dashboard.
+- The dashboard’s “System Health” card is wired to `GET /api/ops/health/status`. When the cached status is missing (HTTP 404), the card displays “No cached health yet” and a call-to-action button. Clicking that button hits the guarded `/api/ops/health/run`, populates the cache, and re-renders the badge/summary once the data returns.
+
+### Gate Error Codes
+
+The unified gate framework uses standardized codes in each gate's `code` field:
+
+- Harvest: `NO_HARVESTED_FILES`, `HARVEST_RANGE_EMPTY`, `HARVEST_STALE`, `HARVEST_AGE_UNKNOWN`
+- Fidelity: `FIDELITY_MISSING`, `FIDELITY_WARNING`, `FIDELITY_UNTRUSTED`, `FIDELITY_UNKNOWN`
+- Calibration: `CALIBRATION_MISSING`, `CALIBRATION_FAILED`, `CALIBRATION_STALE`, `CALIBRATION_BLOCKED`, `CALIBRATION_AGE_UNKNOWN`
+
+---
+
+## Backtest Preflight (fail-fast)
+
+`POST /api/backtest/start` performs a **preflight** before spawning the backtest worker.
+
+- If the backtest is **historical** and uses `chain_mode=live_chain` (the default for historical), the system requires harvested snapshots under `data/live_deribit/*/*.parquet`.
+- Preflight failures return a canonical error envelope:
+
+```json
+{
+  "ok": false,
+  "error": {
+    "code": "NO_HARVESTED_FILES",
+    "message": "No harvested files available for requested date range.",
+    "details": {
+      "data_readiness": {
+        "harvest_required": true,
+        "harvest": {"available": false},
+        "fidelity": {"available": false},
+        "calibration": {"available": false}
+      },
+      "gates": [
+        {"name": "harvest", "mode": "block", "status": "FAIL", "code": "NO_HARVESTED_FILES", "message": "No harvested files available."}
+      ],
+      "gate_overall": {"status": "FAIL", "severity": "FATAL", "can_trade": false},
+      "effective_config": {"chain_mode": "live_chain", "is_historical": true}
+    }
+  }
+}
+```
+
+Preflight also enforces the optional fidelity gate (`FIDELITY_GATE_MODE=warn|block`) without spawning workers.
+
+## Smoke Tests (manual after changes)
+
+Smoke tests are manual and are not substitutes for the operational health checks above.
 
 ### 1. Live Agent Dry-Run Test
 
diff --git a/agent_loop.py b/agent_loop.py
index 673ea94..3e96631 100644
--- a/agent_loop.py
+++ b/agent_loop.py
@@ -36,6 +36,7 @@ from src.healthcheck import (
     run_and_cache_healthcheck,
     set_agent_paused_due_to_health,
     is_agent_paused_due_to_health,
+    get_cached_health_status,
 )
 from src.deribit.base_client import HealthSeverity
 
@@ -45,6 +46,18 @@ shutdown_requested = False
 last_health_recheck_time: float = 0
 
 
+def _health_trading_allowed() -> tuple[bool, str]:
+    """Return (allowed, reason) for trading based on cached health."""
+    cached = get_cached_health_status()
+    if cached is None:
+        return False, "missing_cached_health"
+    if cached.can_trade is False:
+        severity = cached.worst_severity or "unknown"
+        reason = cached.summary or "can_trade=False"
+        return False, f"blocked_by_health can_trade=False severity={severity}: {reason}"
+    return True, ""
+
+
 def signal_handler(signum: int, frame: object) -> None:
     """Handle shutdown signals gracefully."""
     global shutdown_requested
@@ -302,6 +315,14 @@ def run_agent_loop_forever(
             print(f"Iteration {iteration} - {datetime.utcnow().isoformat()}")
             print(f"{'='*60}")
             
+            health_allowed, health_reason = _health_trading_allowed()
+            if not health_allowed:
+                if not is_agent_paused_due_to_health():
+                    set_agent_paused_due_to_health(True)
+                print(f"\n[HEALTH GUARD] blocked_by_health ({health_reason}). Skipping trading.")
+                time.sleep(settings.loop_interval_sec)
+                continue
+
             if is_agent_paused_due_to_health():
                 print("\n[HEALTH GUARD] Agent paused due to health failure. Skipping trading.")
                 print("[HEALTH GUARD] Will re-check health on next interval.")
diff --git a/attached_assets/Pasted-ROLE-You-are-a-senior-product-minded-engineer-Update-th_1766256235012.txt b/attached_assets/Pasted-ROLE-You-are-a-senior-product-minded-engineer-Update-th_1766256235012.txt
deleted file mode 100644
index 41e79f8..0000000
--- a/attached_assets/Pasted-ROLE-You-are-a-senior-product-minded-engineer-Update-th_1766256235012.txt
+++ /dev/null
@@ -1,73 +0,0 @@
-ROLE
-You are a senior product-minded engineer. Update the repo’s roadmap file to reflect the new North Star discovered during Synthetic Fidelity + live replay work.
-
-FILE TO EDIT
-- ROADMAP_BACKLOG.md (if the repo uses a different name, locate the exact roadmap file; in this workspace it’s “ROADMAP_BACKLOG (2).md” content)
-
-CONTEXT YOU MUST PRESERVE
-The current roadmap already contains:
-- Phases 1–3 at the top
-- A strong section [E1.5] “Synthetic Fidelity Score + trading gate”
-- Recently Completed includes: synthetic regimes, live_deribit datasource, compare/diff modules, backtest lab UI, shared state_core, etc.
-
-You must keep the existing content, but REFRAME the plan so Fidelity Gate + measurement integrity are the organizing principle.
-
-GOALS
-1) Update the North Star / Phase plan to add a new Phase 0 (Truth Foundation + Fidelity Gate).
-2) Promote E1.5 from “one backlog item” into the prime directive, with enforcement language and clear exit criteria.
-3) Add a new P0 backlog section: “Measurement Integrity & Data Contracts” with concrete items.
-4) Ensure priorities are explicit: Truth → Trust → Trade, then scale strategies/bots, then SaaS.
-
-EDIT SPEC (DETAILED)
-
-A) Update the “Phases:” block near the top
-Replace the existing Phase list with:
-
-- Phase 0 – Truth Foundation + Fidelity Gate (new, must be complete before trusting backtests)
-- Phase 1 – One good covered-call bot on testnet with gate enforced (mostly done, but now “done” means TRUSTED)
-- Phase 2 – Strategy Packs (GregBot + others) with deterministic replay + parity
-- Phase 3 – Multi-bot supervisor + real historical data + production ops + heavy quant research
-
-Add a short “Definition of Done” bullet under the Phase list:
-- A strategy is “done” only if deterministic replay + audited PnL + passes Fidelity Gate.
-
-B) Add a new section near the top (after “Recently Completed” and before “A. Architecture & Design”)
-Title: “0. North Star: Truth → Trust → Trade”
-Include:
-- Why: synthetic backtests are untrusted unless fidelity gate is green
-- The gate labels: TRUSTED / WARNING / UNTRUSTED
-- Promotion ladder: synthetic → live replay → paper → testnet → mainnet
-- Hard rule: no auto-promotion; fidelity must pass
-
-C) Add a new P0 section (place it under “B. Persistence & Infrastructure” or create a new section “B0. Measurement Integrity & Data Contracts”)
-Items (P0):
-- Enforce premium units contract (option premiums are USD inside backtester; live_deribit premiums must be converted from underlying units)
-- Add invariant tests:
-  - drawdown sanity (non-negative, bounded in normal runs)
-  - PnL unit sanity (premium_usd vs underlying_price)
-- Document the contract in-doc: “units at each boundary” (harvester → exam → datasource → simulator)
-
-D) Update [E1.5] section
-Keep the existing content, but add:
-- “Enforcement” subsection:
-  - backtest UI must show gate label
-  - optimizer/strategy factory blocked unless TRUSTED (or tagged “exploratory-only”)
-  - live trading blocked unless latest calibration bundle is TRUSTED and not stale
-- “Exit criteria” subsection:
-  - min coverage threshold
-  - min strategy parity threshold for canonical strategies
-  - max allowed IV bucket MAE / vega-weighted MAE
-
-E) Add a small note under “G2 Backtest Lab enhancements” that fidelity badges must be shown
-(Do not build UI here; just make it explicit in roadmap that the backtest lab must surface gate status.)
-
-STYLE REQUIREMENTS
-- Keep the existing headings/IDs (A1, A2, E1.5, etc.) intact.
-- Add new sections without deleting old ones.
-- Be concise: roadmap is a control document, not an essay.
-
-ACCEPTANCE CRITERIA
-- Roadmap has Phase 0 added and clearly defines “done”
-- E1.5 explicitly states enforcement (gate blocks trust/trading)
-- New P0 “Measurement Integrity & Data Contracts” exists and is prioritized above strategy expansion
-- No loss of existing backlog items
diff --git a/data/backtests/index.jsonl b/data/backtests/index.jsonl
index 1b70a31..54d1939 100644
--- a/data/backtests/index.jsonl
+++ b/data/backtests/index.jsonl
@@ -1,3 +1,21 @@
+{"run_id": "2025-12-20T18-00-49Z_BTC_8696f53d", "created_at": "2025-12-20T18:00:49.037207+00:00", "underlying": "BTC", "start_date": "2025-12-07", "end_date": "2025-12-13", "status": "finished", "primary_exit_style": "tp_and_roll", "net_profit_pct": 17.8141689, "max_drawdown_pct": 69.74500725391692, "sharpe_ratio": 0.0, "num_trades": 8}
+{"run_id": "2025-12-20T18-00-44Z_BTC_6f0d5259", "created_at": "2025-12-20T18:00:44.828734+00:00", "underlying": "BTC", "start_date": "2025-12-07", "end_date": "2025-12-13", "status": "finished", "primary_exit_style": "tp_and_roll", "net_profit_pct": 15.520456718055934, "max_drawdown_pct": 75.75119335301032, "sharpe_ratio": 0.0, "num_trades": 8}
+{"run_id": "2025-12-20T18-00-42Z_BTC_aca321ad", "created_at": "2025-12-20T18:00:42.294581+00:00", "underlying": "BTC", "start_date": "2025-12-07", "end_date": "2025-12-13", "status": "finished", "primary_exit_style": "hold_to_expiry", "net_profit_pct": -96.8166285098635, "max_drawdown_pct": 732.1046240677935, "sharpe_ratio": 0.0, "num_trades": 8}
+{"run_id": "2025-12-20T18-00-38Z_BTC_872794f5", "created_at": "2025-12-20T18:00:38.743247+00:00", "underlying": "BTC", "start_date": "2025-12-07", "end_date": "2025-12-13", "status": "finished", "primary_exit_style": "hold_to_expiry", "net_profit_pct": -91.98114545064861, "max_drawdown_pct": 709.7279509351115, "sharpe_ratio": 0.0, "num_trades": 8}
+{"run_id": "2025-12-20T17-48-26Z_BTC_9ffec9e8", "created_at": "2025-12-20T17:48:26.991459+00:00", "underlying": "BTC", "start_date": "2025-12-10", "end_date": "2025-12-13", "status": "finished", "primary_exit_style": "tp_and_roll", "net_profit_pct": 57.59303349999999, "max_drawdown_pct": 0.0, "sharpe_ratio": 0.0, "num_trades": 20}
+{"run_id": "2025-12-20T17-48-23Z_BTC_bd092e41", "created_at": "2025-12-20T17:48:23.656234+00:00", "underlying": "BTC", "start_date": "2025-12-10", "end_date": "2025-12-13", "status": "finished", "primary_exit_style": "tp_and_roll", "net_profit_pct": 62.782024643397435, "max_drawdown_pct": 0.0, "sharpe_ratio": 0.0, "num_trades": 20}
+{"run_id": "2025-12-20T17-48-20Z_BTC_1c7aece6", "created_at": "2025-12-20T17:48:20.169591+00:00", "underlying": "BTC", "start_date": "2025-12-10", "end_date": "2025-12-13", "status": "finished", "primary_exit_style": "hold_to_expiry", "net_profit_pct": -79.67056649999998, "max_drawdown_pct": 10926.100180886597, "sharpe_ratio": 0.0, "num_trades": 20}
+{"run_id": "2025-12-20T17-48-16Z_BTC_a04db4cb", "created_at": "2025-12-20T17:48:16.050770+00:00", "underlying": "BTC", "start_date": "2025-12-10", "end_date": "2025-12-13", "status": "finished", "primary_exit_style": "hold_to_expiry", "net_profit_pct": -74.48157535660285, "max_drawdown_pct": 5301.185246651486, "sharpe_ratio": 0.0, "num_trades": 20}
+{"run_id": "2025-12-20T17-46-15Z_BTC_7a9fcec4", "created_at": "2025-12-20T17:46:15.098090+00:00", "underlying": "BTC", "start_date": "2025-12-10", "end_date": "2025-12-13", "status": "finished", "primary_exit_style": "tp_and_roll", "net_profit_pct": 0.0, "max_drawdown_pct": 0.0, "sharpe_ratio": 0.0, "num_trades": 0}
+{"run_id": "2025-12-20T17-46-11Z_BTC_61c04c6b", "created_at": "2025-12-20T17:46:11.806403+00:00", "underlying": "BTC", "start_date": "2025-12-10", "end_date": "2025-12-13", "status": "finished", "primary_exit_style": "tp_and_roll", "net_profit_pct": 10.129996053522591, "max_drawdown_pct": 20.785119319814665, "sharpe_ratio": 0.0, "num_trades": 12}
+{"run_id": "2025-12-20T17-46-08Z_BTC_17c49061", "created_at": "2025-12-20T17:46:08.481364+00:00", "underlying": "BTC", "start_date": "2025-12-10", "end_date": "2025-12-13", "status": "finished", "primary_exit_style": "hold_to_expiry", "net_profit_pct": 0.0, "max_drawdown_pct": 0.0, "sharpe_ratio": 0.0, "num_trades": 0}
+{"run_id": "2025-12-20T17-46-04Z_BTC_d9fdfdc1", "created_at": "2025-12-20T17:46:04.487702+00:00", "underlying": "BTC", "start_date": "2025-12-10", "end_date": "2025-12-13", "status": "finished", "primary_exit_style": "hold_to_expiry", "net_profit_pct": -11.156403946477367, "max_drawdown_pct": 846.6285195195594, "sharpe_ratio": 0.0, "num_trades": 12}
+{"run_id": "2025-12-20T17-44-18Z_BTC_3c3acec3", "created_at": "2025-12-20T17:44:18.961916+00:00", "underlying": "BTC", "start_date": "2025-12-10", "end_date": "2025-12-13", "status": "failed", "primary_exit_style": "tp_and_roll", "net_profit_pct": 0.0, "max_drawdown_pct": 0.0, "sharpe_ratio": 0.0, "num_trades": 0, "error": "[build_historical_state] Both live_chain and synthetic_grid returned empty candidates at 2025-12-10 19:00:00+00:00 for BTC. sigma=0.1000, spot=92938.54, DTE range=[1, 21]"}
+{"run_id": "2025-12-20T17-44-16Z_BTC_5f1c7ed7", "created_at": "2025-12-20T17:44:16.034027+00:00", "underlying": "BTC", "start_date": "2025-12-10", "end_date": "2025-12-13", "status": "failed", "primary_exit_style": "hold_to_expiry", "net_profit_pct": 0.0, "max_drawdown_pct": 0.0, "sharpe_ratio": 0.0, "num_trades": 0, "error": "[build_historical_state] Both live_chain and synthetic_grid returned empty candidates at 2025-12-10 19:00:00+00:00 for BTC. sigma=0.1000, spot=92938.54, DTE range=[1, 21]"}
+{"run_id": "2025-12-20T17-35-40Z_BTC_7f53a08a", "created_at": "2025-12-20T17:35:40.990440+00:00", "underlying": "BTC", "start_date": "2025-12-10", "end_date": "2025-12-13", "status": "finished", "primary_exit_style": "tp_and_roll", "net_profit_pct": 0, "max_drawdown_pct": 0.0, "sharpe_ratio": 0, "num_trades": 0}
+{"run_id": "2025-12-20T17-35-40Z_BTC_f98837c0", "created_at": "2025-12-20T17:35:40.979393+00:00", "underlying": "BTC", "start_date": "2025-12-10", "end_date": "2025-12-13", "status": "finished", "primary_exit_style": "tp_and_roll", "net_profit_pct": 0, "max_drawdown_pct": 0.0, "sharpe_ratio": 0, "num_trades": 0}
+{"run_id": "2025-12-20T17-35-40Z_BTC_149b1f38", "created_at": "2025-12-20T17:35:40.967545+00:00", "underlying": "BTC", "start_date": "2025-12-10", "end_date": "2025-12-13", "status": "finished", "primary_exit_style": "hold_to_expiry", "net_profit_pct": 0, "max_drawdown_pct": 0.0, "sharpe_ratio": 0, "num_trades": 0}
+{"run_id": "2025-12-20T17-35-40Z_BTC_c7f1ed8c", "created_at": "2025-12-20T17:35:40.949641+00:00", "underlying": "BTC", "start_date": "2025-12-10", "end_date": "2025-12-13", "status": "finished", "primary_exit_style": "hold_to_expiry", "net_profit_pct": 0, "max_drawdown_pct": 0.0, "sharpe_ratio": 0, "num_trades": 0}
 {"run_id": "2025-12-18T22-57-20Z_BTC_4b2651a4", "created_at": "2025-12-18T22:57:20.501665", "underlying": "BTC", "start_date": "2024-01-01T00:00:00+00:00", "end_date": "2024-01-07T00:00:00+00:00", "status": "finished", "primary_exit_style": "hold_to_expiry", "net_profit_pct": 0.0, "max_drawdown_pct": 0.0, "sharpe_ratio": 0.0, "num_trades": 0}
 {"run_id": "2025-12-18T21-07-07Z_BTC_506dfd07", "created_at": "2025-12-18T21:07:07.366533", "underlying": "BTC", "start_date": "2024-01-01T00:00:00+00:00", "end_date": "2024-01-07T00:00:00+00:00", "status": "finished", "primary_exit_style": "hold_to_expiry", "net_profit_pct": 0.0, "max_drawdown_pct": 0.0, "sharpe_ratio": 0.0, "num_trades": 0}
 {"run_id": "2025-12-18T19-08-51Z_BTC_5fdeffcb", "created_at": "2025-12-18T19:08:51.281554", "underlying": "BTC", "start_date": "2024-01-01T00:00:00+00:00", "end_date": "2024-01-07T00:00:00+00:00", "status": "finished", "primary_exit_style": "hold_to_expiry", "net_profit_pct": 0.0, "max_drawdown_pct": 0.0, "sharpe_ratio": 0.0, "num_trades": 0}
diff --git a/data/fidelity_runs/BTC/history/20251219_234717/fidelity_report.json b/data/fidelity_runs/BTC/history/20251219_234717/fidelity_report.json
deleted file mode 100644
index 135e9e4..0000000
--- a/data/fidelity_runs/BTC/history/20251219_234717/fidelity_report.json
+++ /dev/null
@@ -1,131 +0,0 @@
-{
-  "component_scores": {
-    "strategy_pnl_parity": 100.0,
-    "underlying_returns": 100.0
-  },
-  "gate": "TRUSTED",
-  "market_live_meta": {
-    "ds_class": "LiveDeribitDataSource",
-    "margin_type": "linear",
-    "settlement_ccy": "USDC",
-    "type": "live_replay",
-    "underlying": "BTC"
-  },
-  "market_synth_meta": {
-    "cfg_class": "CallSimulationConfig",
-    "type": "synthetic_replay",
-    "underlying": "BTC"
-  },
-  "overall_score": 100.0,
-  "run_id": "20251219_234717",
-  "strategy_parity": {
-    "decision_times": [
-      "2025-12-07T00:00:00+00:00",
-      "2025-12-08T00:00:00+00:00",
-      "2025-12-09T00:00:00+00:00",
-      "2025-12-10T00:00:00+00:00",
-      "2025-12-11T00:00:00+00:00",
-      "2025-12-12T00:00:00+00:00",
-      "2025-12-13T00:00:00+00:00",
-      "2025-12-14T00:00:00+00:00",
-      "2025-12-15T00:00:00+00:00",
-      "2025-12-16T00:00:00+00:00",
-      "2025-12-17T00:00:00+00:00",
-      "2025-12-18T00:00:00+00:00",
-      "2025-12-19T00:00:00+00:00"
-    ],
-    "strategies": [
-      {
-        "live": {
-          "notes": "P0 placeholder (execution not implemented)",
-          "num_trades": 0,
-          "spot_first": 0.0,
-          "spot_last": 0.0
-        },
-        "name": "covered_call",
-        "synthetic": {
-          "notes": "P0 placeholder (execution not implemented)",
-          "num_trades": 0,
-          "spot_first": 0.0,
-          "spot_last": 85758.41
-        }
-      },
-      {
-        "live": {
-          "notes": "P0 placeholder (execution not implemented)",
-          "num_trades": 0,
-          "spot_first": 0.0,
-          "spot_last": 0.0
-        },
-        "name": "cash_secured_put",
-        "synthetic": {
-          "notes": "P0 placeholder (execution not implemented)",
-          "num_trades": 0,
-          "spot_first": 0.0,
-          "spot_last": 85758.41
-        }
-      },
-      {
-        "live": {
-          "notes": "P0 placeholder (execution not implemented)",
-          "num_trades": 0,
-          "spot_first": 0.0,
-          "spot_last": 0.0
-        },
-        "name": "short_strangle",
-        "synthetic": {
-          "notes": "P0 placeholder (execution not implemented)",
-          "num_trades": 0,
-          "spot_first": 0.0,
-          "spot_last": 85758.41
-        }
-      },
-      {
-        "live": {
-          "notes": "P0 placeholder (execution not implemented)",
-          "num_trades": 0,
-          "spot_first": 0.0,
-          "spot_last": 0.0
-        },
-        "name": "put_spread_credit",
-        "synthetic": {
-          "notes": "P0 placeholder (execution not implemented)",
-          "num_trades": 0,
-          "spot_first": 0.0,
-          "spot_last": 85758.41
-        }
-      },
-      {
-        "live": {
-          "notes": "P0 placeholder (execution not implemented)",
-          "num_trades": 0,
-          "spot_first": 0.0,
-          "spot_last": 0.0
-        },
-        "name": "call_spread_debit",
-        "synthetic": {
-          "notes": "P0 placeholder (execution not implemented)",
-          "num_trades": 0,
-          "spot_first": 0.0,
-          "spot_last": 85758.41
-        }
-      },
-      {
-        "live": {
-          "notes": "P0 placeholder (execution not implemented)",
-          "num_trades": 0,
-          "spot_first": 0.0,
-          "spot_last": 0.0
-        },
-        "name": "calendar",
-        "synthetic": {
-          "notes": "P0 placeholder (execution not implemented)",
-          "num_trades": 0,
-          "spot_first": 0.0,
-          "spot_last": 85758.41
-        }
-      }
-    ]
-  },
-  "timestamp": "2025-12-19T23:47:17.593360+00:00"
-}
\ No newline at end of file
diff --git a/data/fidelity_runs/BTC/history/20251219_234717/fidelity_report.md b/data/fidelity_runs/BTC/history/20251219_234717/fidelity_report.md
deleted file mode 100644
index 68f3de9..0000000
--- a/data/fidelity_runs/BTC/history/20251219_234717/fidelity_report.md
+++ /dev/null
@@ -1,147 +0,0 @@
-# Synthetic Fidelity Report
-
-- Run ID: 20251219_234717
-- Timestamp (UTC): 2025-12-19T23:47:17.593360+00:00
-- Gate: **TRUSTED**
-
-## Scores
-
-- Overall: **100.0**
-- strategy_pnl_parity: 100.0
-- underlying_returns: 100.0
-
-## Market Meta
-
-### Live
-```json
-{
-  "ds_class": "LiveDeribitDataSource",
-  "margin_type": "linear",
-  "settlement_ccy": "USDC",
-  "type": "live_replay",
-  "underlying": "BTC"
-}
-```
-
-### Synthetic
-```json
-{
-  "cfg_class": "CallSimulationConfig",
-  "type": "synthetic_replay",
-  "underlying": "BTC"
-}
-```
-
-## Strategy Parity (P0 placeholder)
-
-```json
-{
-  "decision_times": [
-    "2025-12-07T00:00:00+00:00",
-    "2025-12-08T00:00:00+00:00",
-    "2025-12-09T00:00:00+00:00",
-    "2025-12-10T00:00:00+00:00",
-    "2025-12-11T00:00:00+00:00",
-    "2025-12-12T00:00:00+00:00",
-    "2025-12-13T00:00:00+00:00",
-    "2025-12-14T00:00:00+00:00",
-    "2025-12-15T00:00:00+00:00",
-    "2025-12-16T00:00:00+00:00",
-    "2025-12-17T00:00:00+00:00",
-    "2025-12-18T00:00:00+00:00",
-    "2025-12-19T00:00:00+00:00"
-  ],
-  "strategies": [
-    {
-      "live": {
-        "notes": "P0 placeholder (execution not implemented)",
-        "num_trades": 0,
-        "spot_first": 0.0,
-        "spot_last": 0.0
-      },
-      "name": "covered_call",
-      "synthetic": {
-        "notes": "P0 placeholder (execution not implemented)",
-        "num_trades": 0,
-        "spot_first": 0.0,
-        "spot_last": 85758.41
-      }
-    },
-    {
-      "live": {
-        "notes": "P0 placeholder (execution not implemented)",
-        "num_trades": 0,
-        "spot_first": 0.0,
-        "spot_last": 0.0
-      },
-      "name": "cash_secured_put",
-      "synthetic": {
-        "notes": "P0 placeholder (execution not implemented)",
-        "num_trades": 0,
-        "spot_first": 0.0,
-        "spot_last": 85758.41
-      }
-    },
-    {
-      "live": {
-        "notes": "P0 placeholder (execution not implemented)",
-        "num_trades": 0,
-        "spot_first": 0.0,
-        "spot_last": 0.0
-      },
-      "name": "short_strangle",
-      "synthetic": {
-        "notes": "P0 placeholder (execution not implemented)",
-        "num_trades": 0,
-        "spot_first": 0.0,
-        "spot_last": 85758.41
-      }
-    },
-    {
-      "live": {
-        "notes": "P0 placeholder (execution not implemented)",
-        "num_trades": 0,
-        "spot_first": 0.0,
-        "spot_last": 0.0
-      },
-      "name": "put_spread_credit",
-      "synthetic": {
-        "notes": "P0 placeholder (execution not implemented)",
-        "num_trades": 0,
-        "spot_first": 0.0,
-        "spot_last": 85758.41
-      }
-    },
-    {
-      "live": {
-        "notes": "P0 placeholder (execution not implemented)",
-        "num_trades": 0,
-        "spot_first": 0.0,
-        "spot_last": 0.0
-      },
-      "name": "call_spread_debit",
-      "synthetic": {
-        "notes": "P0 placeholder (execution not implemented)",
-        "num_trades": 0,
-        "spot_first": 0.0,
-        "spot_last": 85758.41
-      }
-    },
-    {
-      "live": {
-        "notes": "P0 placeholder (execution not implemented)",
-        "num_trades": 0,
-        "spot_first": 0.0,
-        "spot_last": 0.0
-      },
-      "name": "calendar",
-      "synthetic": {
-        "notes": "P0 placeholder (execution not implemented)",
-        "num_trades": 0,
-        "spot_first": 0.0,
-        "spot_last": 85758.41
-      }
-    }
-  ]
-}
-```
diff --git a/data/fidelity_runs/BTC/latest/fidelity_report.json b/data/fidelity_runs/BTC/latest/fidelity_report.json
deleted file mode 100644
index 135e9e4..0000000
--- a/data/fidelity_runs/BTC/latest/fidelity_report.json
+++ /dev/null
@@ -1,131 +0,0 @@
-{
-  "component_scores": {
-    "strategy_pnl_parity": 100.0,
-    "underlying_returns": 100.0
-  },
-  "gate": "TRUSTED",
-  "market_live_meta": {
-    "ds_class": "LiveDeribitDataSource",
-    "margin_type": "linear",
-    "settlement_ccy": "USDC",
-    "type": "live_replay",
-    "underlying": "BTC"
-  },
-  "market_synth_meta": {
-    "cfg_class": "CallSimulationConfig",
-    "type": "synthetic_replay",
-    "underlying": "BTC"
-  },
-  "overall_score": 100.0,
-  "run_id": "20251219_234717",
-  "strategy_parity": {
-    "decision_times": [
-      "2025-12-07T00:00:00+00:00",
-      "2025-12-08T00:00:00+00:00",
-      "2025-12-09T00:00:00+00:00",
-      "2025-12-10T00:00:00+00:00",
-      "2025-12-11T00:00:00+00:00",
-      "2025-12-12T00:00:00+00:00",
-      "2025-12-13T00:00:00+00:00",
-      "2025-12-14T00:00:00+00:00",
-      "2025-12-15T00:00:00+00:00",
-      "2025-12-16T00:00:00+00:00",
-      "2025-12-17T00:00:00+00:00",
-      "2025-12-18T00:00:00+00:00",
-      "2025-12-19T00:00:00+00:00"
-    ],
-    "strategies": [
-      {
-        "live": {
-          "notes": "P0 placeholder (execution not implemented)",
-          "num_trades": 0,
-          "spot_first": 0.0,
-          "spot_last": 0.0
-        },
-        "name": "covered_call",
-        "synthetic": {
-          "notes": "P0 placeholder (execution not implemented)",
-          "num_trades": 0,
-          "spot_first": 0.0,
-          "spot_last": 85758.41
-        }
-      },
-      {
-        "live": {
-          "notes": "P0 placeholder (execution not implemented)",
-          "num_trades": 0,
-          "spot_first": 0.0,
-          "spot_last": 0.0
-        },
-        "name": "cash_secured_put",
-        "synthetic": {
-          "notes": "P0 placeholder (execution not implemented)",
-          "num_trades": 0,
-          "spot_first": 0.0,
-          "spot_last": 85758.41
-        }
-      },
-      {
-        "live": {
-          "notes": "P0 placeholder (execution not implemented)",
-          "num_trades": 0,
-          "spot_first": 0.0,
-          "spot_last": 0.0
-        },
-        "name": "short_strangle",
-        "synthetic": {
-          "notes": "P0 placeholder (execution not implemented)",
-          "num_trades": 0,
-          "spot_first": 0.0,
-          "spot_last": 85758.41
-        }
-      },
-      {
-        "live": {
-          "notes": "P0 placeholder (execution not implemented)",
-          "num_trades": 0,
-          "spot_first": 0.0,
-          "spot_last": 0.0
-        },
-        "name": "put_spread_credit",
-        "synthetic": {
-          "notes": "P0 placeholder (execution not implemented)",
-          "num_trades": 0,
-          "spot_first": 0.0,
-          "spot_last": 85758.41
-        }
-      },
-      {
-        "live": {
-          "notes": "P0 placeholder (execution not implemented)",
-          "num_trades": 0,
-          "spot_first": 0.0,
-          "spot_last": 0.0
-        },
-        "name": "call_spread_debit",
-        "synthetic": {
-          "notes": "P0 placeholder (execution not implemented)",
-          "num_trades": 0,
-          "spot_first": 0.0,
-          "spot_last": 85758.41
-        }
-      },
-      {
-        "live": {
-          "notes": "P0 placeholder (execution not implemented)",
-          "num_trades": 0,
-          "spot_first": 0.0,
-          "spot_last": 0.0
-        },
-        "name": "calendar",
-        "synthetic": {
-          "notes": "P0 placeholder (execution not implemented)",
-          "num_trades": 0,
-          "spot_first": 0.0,
-          "spot_last": 85758.41
-        }
-      }
-    ]
-  },
-  "timestamp": "2025-12-19T23:47:17.593360+00:00"
-}
\ No newline at end of file
diff --git a/data/fidelity_runs/BTC/latest/fidelity_report.md b/data/fidelity_runs/BTC/latest/fidelity_report.md
deleted file mode 100644
index 68f3de9..0000000
--- a/data/fidelity_runs/BTC/latest/fidelity_report.md
+++ /dev/null
@@ -1,147 +0,0 @@
-# Synthetic Fidelity Report
-
-- Run ID: 20251219_234717
-- Timestamp (UTC): 2025-12-19T23:47:17.593360+00:00
-- Gate: **TRUSTED**
-
-## Scores
-
-- Overall: **100.0**
-- strategy_pnl_parity: 100.0
-- underlying_returns: 100.0
-
-## Market Meta
-
-### Live
-```json
-{
-  "ds_class": "LiveDeribitDataSource",
-  "margin_type": "linear",
-  "settlement_ccy": "USDC",
-  "type": "live_replay",
-  "underlying": "BTC"
-}
-```
-
-### Synthetic
-```json
-{
-  "cfg_class": "CallSimulationConfig",
-  "type": "synthetic_replay",
-  "underlying": "BTC"
-}
-```
-
-## Strategy Parity (P0 placeholder)
-
-```json
-{
-  "decision_times": [
-    "2025-12-07T00:00:00+00:00",
-    "2025-12-08T00:00:00+00:00",
-    "2025-12-09T00:00:00+00:00",
-    "2025-12-10T00:00:00+00:00",
-    "2025-12-11T00:00:00+00:00",
-    "2025-12-12T00:00:00+00:00",
-    "2025-12-13T00:00:00+00:00",
-    "2025-12-14T00:00:00+00:00",
-    "2025-12-15T00:00:00+00:00",
-    "2025-12-16T00:00:00+00:00",
-    "2025-12-17T00:00:00+00:00",
-    "2025-12-18T00:00:00+00:00",
-    "2025-12-19T00:00:00+00:00"
-  ],
-  "strategies": [
-    {
-      "live": {
-        "notes": "P0 placeholder (execution not implemented)",
-        "num_trades": 0,
-        "spot_first": 0.0,
-        "spot_last": 0.0
-      },
-      "name": "covered_call",
-      "synthetic": {
-        "notes": "P0 placeholder (execution not implemented)",
-        "num_trades": 0,
-        "spot_first": 0.0,
-        "spot_last": 85758.41
-      }
-    },
-    {
-      "live": {
-        "notes": "P0 placeholder (execution not implemented)",
-        "num_trades": 0,
-        "spot_first": 0.0,
-        "spot_last": 0.0
-      },
-      "name": "cash_secured_put",
-      "synthetic": {
-        "notes": "P0 placeholder (execution not implemented)",
-        "num_trades": 0,
-        "spot_first": 0.0,
-        "spot_last": 85758.41
-      }
-    },
-    {
-      "live": {
-        "notes": "P0 placeholder (execution not implemented)",
-        "num_trades": 0,
-        "spot_first": 0.0,
-        "spot_last": 0.0
-      },
-      "name": "short_strangle",
-      "synthetic": {
-        "notes": "P0 placeholder (execution not implemented)",
-        "num_trades": 0,
-        "spot_first": 0.0,
-        "spot_last": 85758.41
-      }
-    },
-    {
-      "live": {
-        "notes": "P0 placeholder (execution not implemented)",
-        "num_trades": 0,
-        "spot_first": 0.0,
-        "spot_last": 0.0
-      },
-      "name": "put_spread_credit",
-      "synthetic": {
-        "notes": "P0 placeholder (execution not implemented)",
-        "num_trades": 0,
-        "spot_first": 0.0,
-        "spot_last": 85758.41
-      }
-    },
-    {
-      "live": {
-        "notes": "P0 placeholder (execution not implemented)",
-        "num_trades": 0,
-        "spot_first": 0.0,
-        "spot_last": 0.0
-      },
-      "name": "call_spread_debit",
-      "synthetic": {
-        "notes": "P0 placeholder (execution not implemented)",
-        "num_trades": 0,
-        "spot_first": 0.0,
-        "spot_last": 85758.41
-      }
-    },
-    {
-      "live": {
-        "notes": "P0 placeholder (execution not implemented)",
-        "num_trades": 0,
-        "spot_first": 0.0,
-        "spot_last": 0.0
-      },
-      "name": "calendar",
-      "synthetic": {
-        "notes": "P0 placeholder (execution not implemented)",
-        "num_trades": 0,
-        "spot_first": 0.0,
-        "spot_last": 85758.41
-      }
-    }
-  ]
-}
-```
diff --git a/scripts/print_fidelity_summary.py b/scripts/print_fidelity_summary.py
new file mode 100644
index 0000000..e6d6411
--- /dev/null
+++ b/scripts/print_fidelity_summary.py
@@ -0,0 +1,94 @@
+#!/usr/bin/env python3
+"""Print a compact, high-signal summary of the latest fidelity run."""
+
+from __future__ import annotations
+
+import json
+from pathlib import Path
+from typing import Any
+
+
+def _load_json(path: Path) -> dict[str, Any]:
+    return json.loads(path.read_text())
+
+
+def main() -> None:
+    latest_path = Path("data/fidelity_runs/latest.json")
+    if not latest_path.exists():
+        raise SystemExit("Missing data/fidelity_runs/latest.json")
+
+    latest = _load_json(latest_path)
+    run_id = latest.get("run_id") or latest.get("latest_run_id") or latest.get("id")
+    if not run_id:
+        raise SystemExit("latest.json missing run id")
+
+    report_path = Path(f"data/fidelity_runs/{run_id}/fidelity_report.json")
+    if not report_path.exists():
+        raise SystemExit(f"Missing report: {report_path}")
+
+    rep = _load_json(report_path)
+
+    def p(label: str, value: Any) -> None:
+        print(f"{label}: {value}")
+
+    print("TOP-LEVEL")
+    p("run_id", rep.get("run_id"))
+    p("timestamp", rep.get("timestamp"))
+    p("underlying", rep.get("underlying"))
+    p("overall_score", rep.get("overall_score"))
+    p("gate_label", rep.get("gate_label"))
+    p("gate_reason", rep.get("gate_reason"))
+    coverage = rep.get("coverage") or {}
+    p("coverage.coverage_ratio", coverage.get("coverage_ratio"))
+    p("coverage.penalty_ratio", coverage.get("penalty_ratio"))
+    p("coverage.total_trades_opened", coverage.get("total_trades_opened"))
+    p("coverage.valid_trades_closed", coverage.get("valid_trades_closed"))
+    p("coverage.invalid_trades_missing_quote", coverage.get("invalid_trades_missing_quote"))
+
+    print("\nREPLAY")
+    rd = rep.get("replay_diagnostics") or {}
+    for side in ("live", "synthetic"):
+        s = rd.get(side) or {}
+        if not s:
+            continue
+        print(f"[{side}]")
+        for k in (
+            "snapshots_count",
+            "spot_min",
+            "spot_max",
+            "spot_avg",
+            "options_count_min",
+            "options_count_max",
+            "options_count_avg",
+        ):
+            if k in s:
+                p(k, s.get(k))
+
+        fs = s.get("first_snapshot") or {}
+        if fs:
+            p("first_snapshot.spot", fs.get("spot"))
+            p("first_snapshot.options_count", fs.get("options_count"))
+            sample = fs.get("sample_options") or []
+            if sample:
+                p("first_snapshot.sample_option_fields", sorted(sample[0].keys()))
+
+    print("\nSTRATEGIES")
+    sd = rep.get("strategy_diagnostics") or {}
+    opened_live = {
+        name: int(((diag or {}).get("live") or {}).get("opened_trades") or 0)
+        for name, diag in sd.items()
+    }
+    top_opened = sorted(opened_live.items(), key=lambda kv: (-kv[1], kv[0]))[:12]
+    p("top_opened_live", top_opened)
+
+    for name, diag in sd.items():
+        live = (diag or {}).get("live") or {}
+        opened = int(live.get("opened_trades") or 0)
+        skips = live.get("skip_reasons") or {}
+        if opened == 0 and skips:
+            top_skips = sorted(skips.items(), key=lambda kv: (-kv[1], kv[0]))[:6]
+            print(f"- {name} top_skips: {top_skips}")
+
+
+if __name__ == "__main__":
+    main()
diff --git a/scripts/run_fidelity_from_lab.py b/scripts/run_fidelity_from_lab.py
new file mode 100644
index 0000000..20cb7f7
--- /dev/null
+++ b/scripts/run_fidelity_from_lab.py
@@ -0,0 +1,56 @@
+#!/usr/bin/env python3
+"""Run the Lab-based Fidelity orchestrator and persist the report.
+
+Usage:
+  PYTHONPATH=. ./.venv/bin/python scripts/run_fidelity_from_lab.py --underlying BTC --start 2025-12-10 --end 2025-12-13
+"""
+
+from __future__ import annotations
+
+import argparse
+from datetime import date, datetime, time, timezone
+
+from src.backtest.fidelity_store import write_fidelity_report
+from src.backtest.fidelity_suite import run_fidelity_from_lab
+
+
+def _parse_date(s: str) -> date:
+    return date.fromisoformat(s)
+
+
+def main() -> None:
+    p = argparse.ArgumentParser()
+    p.add_argument("--underlying", required=True, choices=["BTC", "ETH"], help="Underlying asset")
+    p.add_argument("--start", required=True, help="YYYY-MM-DD (inclusive)")
+    p.add_argument("--end", required=True, help="YYYY-MM-DD (inclusive)")
+    p.add_argument("--decision-interval-minutes", type=int, default=60)
+    p.add_argument("--min-trades-per-case", type=int, default=5)
+    args = p.parse_args()
+
+    start_d = _parse_date(args.start)
+    end_d = _parse_date(args.end)
+
+    start_ts = datetime.combine(start_d, time.min, tzinfo=timezone.utc)
+    # inclusive end date
+    end_ts = datetime.combine(end_d, time.max, tzinfo=timezone.utc)
+
+    report = run_fidelity_from_lab(
+        underlying=args.underlying,
+        start_ts=start_ts,
+        end_ts=end_ts,
+        decision_interval_minutes=int(args.decision_interval_minutes),
+        min_trades_per_case=int(args.min_trades_per_case),
+    )
+
+    summary = write_fidelity_report(report)
+
+    run_id = summary.get("run_id")
+    base_dir = "data/fidelity_runs"
+    print(f"run_id={run_id}")
+    print(f"overall_score={summary.get('overall_score')}")
+    print(f"gate_label={summary.get('gate_label')}")
+    print(f"report_path={base_dir}/{run_id}/report.json")
+
+
+if __name__ == "__main__":
+    main()
diff --git a/scripts/run_fidelity_from_lab_daily.py b/scripts/run_fidelity_from_lab_daily.py
new file mode 100644
index 0000000..5420368
--- /dev/null
+++ b/scripts/run_fidelity_from_lab_daily.py
@@ -0,0 +1,56 @@
+#!/usr/bin/env python3
+"""Daily automation helper for Lab-based Synthetic Fidelity.
+
+This is intended for local cron / operator runs (not CI), since it relies on
+harvested live_deribit data being present.
+
+Default behavior:
+- Runs BTC and ETH for the last 3 full days ending yesterday.
+- Writes reports to the fidelity store (data/fidelity_runs by default).
+
+Usage:
+  PYTHONPATH=. ./.venv/bin/python scripts/run_fidelity_from_lab_daily.py
+  PYTHONPATH=. ./.venv/bin/python scripts/run_fidelity_from_lab_daily.py --days 7
+"""
+
+from __future__ import annotations
+
+import argparse
+from datetime import datetime, timedelta, timezone
+
+from src.backtest.fidelity_store import write_fidelity_report
+from src.backtest.fidelity_suite import run_fidelity_from_lab
+
+
+def main() -> None:
+    p = argparse.ArgumentParser()
+    p.add_argument("--days", type=int, default=3, help="Number of full days to include (ending yesterday)")
+    p.add_argument("--min-trades-per-case", type=int, default=5)
+    p.add_argument("--decision-interval-minutes", type=int, default=60)
+    args = p.parse_args()
+
+    if args.days <= 0:
+        raise SystemExit("--days must be >= 1")
+
+    today = datetime.now(timezone.utc).date()
+    end_date = today - timedelta(days=1)
+    start_date = end_date - timedelta(days=int(args.days) - 1)
+
+    start_ts = datetime(start_date.year, start_date.month, start_date.day, 0, 0, 0, tzinfo=timezone.utc)
+    end_ts = datetime(end_date.year, end_date.month, end_date.day, 23, 59, 59, tzinfo=timezone.utc)
+
+    for underlying in ("BTC", "ETH"):
+        print(f"Running lab fidelity: {underlying} {start_date}..{end_date}")
+        report = run_fidelity_from_lab(
+            underlying=underlying,
+            start_ts=start_ts,
+            end_ts=end_ts,
+            decision_interval_minutes=int(args.decision_interval_minutes),
+            min_trades_per_case=int(args.min_trades_per_case),
+        )
+        summary = write_fidelity_report(report)
+        print(f"  -> run_id={summary.get('run_id')} gate={summary.get('gate_label')} score={summary.get('overall_score')}")
+
+
+if __name__ == "__main__":
+    main()
diff --git a/scripts/sabotage_fidelity_drill.py b/scripts/sabotage_fidelity_drill.py
new file mode 100644
index 0000000..f632755
--- /dev/null
+++ b/scripts/sabotage_fidelity_drill.py
@@ -0,0 +1,42 @@
+#!/usr/bin/env python3
+"""Sabotage drill for Synthetic Fidelity scoring.
+
+This does NOT run real backtests.
+It creates synthetic diff payloads and demonstrates that the scoring function
+moves in the expected direction as diffs get worse.
+
+Usage:
+  PYTHONPATH=. ./.venv/bin/python scripts/sabotage_fidelity_drill.py
+"""
+
+from __future__ import annotations
+
+from src.backtest.fidelity_suite import score_case_from_diff
+
+
+def _payload(net_profit_pp: float, dd_pp: float) -> dict:
+    return {
+        "metrics": {
+            "net_profit_pct": {"diff": net_profit_pp},
+            "max_drawdown_pct": {"diff": dd_pp},
+            "win_rate": {"diff": 0.0},
+            "profit_factor": {"diff": 0.0},
+            "avg_trade_usd": {"diff": 0.0},
+        }
+    }
+
+
+def main() -> None:
+    good, _ = score_case_from_diff(_payload(net_profit_pp=1.0, dd_pp=1.0))
+    bad, _ = score_case_from_diff(_payload(net_profit_pp=20.0, dd_pp=20.0))
+
+    print("Sabotage drill")
+    print(f"  score(good diffs)={good:.2f}")
+    print(f"  score(bad diffs) ={bad:.2f}")
+
+    if good <= bad:
+        raise SystemExit("Expected sabotage to lower the score")
+
+
+if __name__ == "__main__":
+    main()
diff --git a/src/backtest/compare.py b/src/backtest/compare.py
index c574e3d..77f4d90 100644
--- a/src/backtest/compare.py
+++ b/src/backtest/compare.py
@@ -1,20 +1,18 @@
-"""
-Reusable comparison logic for SYNTHETIC vs LIVE_DERIBIT backtests.
+"""Reusable comparison logic for SYNTHETIC vs LIVE_DERIBIT backtests.
+
+This module intentionally reuses the same simulator and file-based storage
+conventions as the Backtest Lab (see src/backtest/manager.py + src/backtest/run_store.py).
+
+It is DB-free by design so it can run in environments without DATABASE_URL.
 """
 
-from datetime import datetime
-from typing import Tuple, Optional, Dict, Any
+from datetime import datetime, timedelta
+from typing import Any, Dict, Optional, Tuple
 
-from src.db import get_db_session
-from src.db.backtest_service import (
-    create_backtest_run,
-    complete_run,
-    fail_run,
-    get_run_with_details,
-)
 from src.backtest.config_schema import DataSourceType
 from src.backtest.covered_call_simulator import CoveredCallSimulator
 from src.backtest.deribit_data_source import DeribitDataSource
+from src.backtest.run_store import create_run, save_run_result, update_run_status
 from src.backtest.types import CallSimulationConfig
 
 
@@ -45,82 +43,153 @@ def run_backtest_with_data_source(
     Raises:
         Exception if backtest fails
     """
-    with get_db_session() as db:
-        try:
-            run = create_backtest_run(
-                db=db,
-                underlying=underlying,
-                start_ts=start_ts,
-                end_ts=end_ts,
-                data_source=data_source.value,
-                decision_interval_minutes=decision_interval_minutes,
-                primary_exit_style=exit_style,
-                config_json={
-                    "underlying": underlying,
-                    "data_source": data_source.value,
-                    "decision_interval_minutes": decision_interval_minutes,
-                    "exit_style": exit_style,
-                },
-            )
-            
-            if verbose:
-                print(f"  Created run: {run.run_id} (data_source={data_source.value})")
-            
-            decision_interval_hours = decision_interval_minutes / 60
-            decision_interval_bars = max(1, int(decision_interval_hours))
-            
-            pricing_mode = "deribit_live" if data_source == DataSourceType.LIVE_DERIBIT else "synthetic_bs"
+    run_result = create_run(
+        {
+            "underlying": underlying,
+            "start_date": start_ts.date().isoformat(),
+            "end_date": end_ts.date().isoformat(),
+            "decision_interval_minutes": decision_interval_minutes,
+            "exit_style": exit_style,
+            "data_source": data_source.value,
+        }
+    )
+    run_id = run_result.run_id
+    update_run_status(run_id, "running")
+
+    if verbose:
+        print(f"  Created run: {run_id} (data_source={data_source.value})")
             
-            config = CallSimulationConfig(
-                underlying=underlying,
-                start=start_ts,
-                end=end_ts,
-                timeframe="1h",
-                decision_interval_bars=decision_interval_bars,
-                initial_spot_position=1.0,
-                contract_size=1.0,
-                fee_rate=0.0003,
-                target_dte=7,
-                dte_tolerance=3,
-                target_delta=0.25,
-                delta_tolerance=0.10,
-                min_dte=1,
-                max_dte=21,
-                delta_min=0.10,
-                delta_max=0.40,
-                option_margin_type="linear",
-                option_settlement_ccy="USDC",
-                tp_threshold_pct=80.0,
-                min_score_to_trade=3.0,
-                pricing_mode=pricing_mode,
+    try:
+        decision_interval_hours = decision_interval_minutes / 60
+        decision_interval_bars = max(1, int(decision_interval_hours))
+
+        pricing_mode = "deribit_live" if data_source == DataSourceType.LIVE_DERIBIT else "synthetic_bs"
+
+        # For small harvested windows (e.g. a few days), a 7DTE strategy produces
+        # zero decision points because the simulator stops at end - target_dte.
+        # Adapt target_dte to the available window while keeping it bounded.
+        window_days = max(1, int((end_ts - start_ts).total_seconds() / 86400))
+        target_dte = max(1, min(7, window_days // 2))
+        dte_tolerance = max(1, min(3, target_dte))
+
+        config = CallSimulationConfig(
+            underlying=underlying,
+            start=start_ts,
+            end=end_ts,
+            timeframe="1h",
+            decision_interval_bars=decision_interval_bars,
+            initial_spot_position=1.0,
+            contract_size=1.0,
+            fee_rate=0.0003,
+            target_dte=target_dte,
+            dte_tolerance=dte_tolerance,
+            target_delta=0.25,
+            delta_tolerance=0.10,
+            min_dte=1,
+            max_dte=21,
+            delta_min=0.10,
+            delta_max=0.40,
+            option_margin_type="linear",
+            option_settlement_ccy="USDC",
+            tp_threshold_pct=80.0,
+            # Fidelity wants measurability; we intentionally trade whenever we have candidates.
+            min_score_to_trade=0.0,
+            pricing_mode=pricing_mode,
+            # Align synthetic runs to the harvested chain universe when available.
+            chain_mode="live_chain",
+            sigma_mode="mark_iv_x_multiplier",
+            synthetic_iv_multiplier=1.0,
+        )
+
+        if data_source in (DataSourceType.LIVE_DERIBIT, DataSourceType.SYNTHETIC):
+            from src.backtest.live_deribit_data_source import LiveDeribitDataSource
+
+            underlying_dir = underlying if "_USDC" in underlying else f"{underlying}_USDC"
+            data_src = LiveDeribitDataSource(
+                underlying=underlying_dir,
+                start_date=start_ts.date(),
+                end_date=end_ts.date(),
+                canonical_underlying=underlying,
             )
-            
-            if data_source == DataSourceType.LIVE_DERIBIT:
-                from src.backtest.live_deribit_data_source import LiveDeribitDataSource
-                
-                data_src = LiveDeribitDataSource(
+        else:
+            data_src = DeribitDataSource()
+
+        simulator = CoveredCallSimulator(data_source=data_src, config=config)
+
+        from src.backtest.state_builder import build_historical_state
+
+        # Prefer decision times aligned to harvested snapshots (when available) so
+        # list_option_chain/get_option_ohlc have data and we don't rely on fallbacks.
+        decision_times = simulator._generate_decision_times()
+        if hasattr(data_src, "get_dataframe"):
+            try:
+                import pandas as pd  # local import to keep module load light
+
+                df = data_src.get_dataframe()
+                if df is not None and (not df.empty) and "harvest_time" in df.columns:
+                    cutoff = end_ts - timedelta(days=target_dte)
+                    raw_times = sorted(pd.to_datetime(df["harvest_time"], utc=True).unique())
+                    snap_times: list[datetime] = []
+                    for ts in raw_times:
+                        if hasattr(ts, "to_pydatetime"):
+                            snap_times.append(ts.to_pydatetime())
+                        else:
+                            snap_times.append(ts)
+                    snap_times = [t for t in snap_times if start_ts <= t <= cutoff]
+                    # Downsample to roughly the requested decision interval.
+                    selected: list[datetime] = []
+                    last: Optional[datetime] = None
+                    min_step = int(decision_interval_minutes) * 60
+                    for t in snap_times:
+                        if last is None or (t - last).total_seconds() >= min_step:
+                            selected.append(t)
+                            last = t
+                    if selected:
+                        decision_times = selected
+            except Exception:
+                # Fall back to regular time grid.
+                pass
+
+        def state_builder(t: datetime):
+            try:
+                return build_historical_state(data_src, config, t)
+            except Exception as e:
+                # For fidelity runs we prefer "skip this decision point" over
+                # aborting the entire compare run.
+                spot_df = data_src.get_spot_ohlc(
                     underlying=underlying,
-                    start_date=start_ts.date(),
-                    end_date=end_ts.date(),
+                    start=t - timedelta(hours=24),
+                    end=t,
+                    timeframe="1h",
                 )
-            else:
-                data_src = DeribitDataSource()
-            
-            simulator = CoveredCallSimulator(data_source=data_src, config=config)
-            
-            def always_trade_policy(candidates, state):
-                return True
-            
-            result = simulator.simulate_policy(policy=always_trade_policy, size=1.0)
-            
-            trades = result.trades if hasattr(result, 'trades') else []
-            metrics = result.metrics if hasattr(result, 'metrics') else {}
-            
-            chains_list = []
-            for trade in trades:
-                chain = getattr(trade, "chain", None)
-                if chain:
-                    chains_list.append({
+                spot = float(spot_df["close"].iloc[-1]) if not spot_df.empty else None
+                return {
+                    "time": t,
+                    "spot": spot,
+                    "underlying": underlying,
+                    "market_context": {},
+                    "candidate_options": [],
+                    "portfolio": {"spot_position": config.initial_spot_position, "equity_usd": None},
+                    "provenance": {"error": str(e)},
+                }
+
+        result = simulator.simulate_policy_with_scoring(
+            decision_times=decision_times,
+            state_builder=state_builder,
+            exit_style=exit_style,
+            min_score_to_trade=config.min_score_to_trade,
+            size=1.0,
+        )
+
+        trades = result.trades if hasattr(result, "trades") else []
+        metrics = result.metrics if hasattr(result, "metrics") else {}
+
+        chains_list = []
+        for trade in trades:
+            chain = getattr(trade, "chain", None)
+            if chain:
+                chains_list.append(
+                    {
                         "open_time": chain.decision_time.isoformat(),
                         "instrument_name": getattr(chain, "instrument_name", None),
                         "num_legs": len(getattr(chain, "legs", [])),
@@ -128,44 +197,57 @@ def run_backtest_with_data_source(
                         "pnl": float(chain.total_pnl),
                         "pnl_vs_hodl": float(getattr(chain, "pnl_vs_hodl", 0)),
                         "max_drawdown_pct": float(chain.max_drawdown_pct),
-                    })
-            
-            formatted_metrics = {
-                "initial_equity": metrics.get("initial_equity", 0),
-                "final_equity": metrics.get("final_equity", 0),
-                "net_profit_usd": metrics.get("final_pnl", 0),
-                "net_profit_pct": metrics.get("total_return_pct", 0),
-                "max_drawdown_pct": metrics.get("max_drawdown_pct", 0),
-                "num_trades": metrics.get("num_trades", 0),
-                "win_rate": metrics.get("win_rate", 0) * 100 if metrics.get("win_rate") else 0,
-                "sharpe_ratio": metrics.get("sharpe_ratio", 0),
-                "sortino_ratio": metrics.get("sortino_ratio", 0),
-                "profit_factor": metrics.get("profit_factor", 0),
-                "final_pnl_vs_hodl": metrics.get("total_pnl_vs_hodl", 0),
-            }
-            
-            metrics_by_style = {exit_style: formatted_metrics}
-            chains_by_style = {exit_style: chains_list}
-            
-            complete_run(
-                db=db,
-                run=run,
-                metrics_by_style=metrics_by_style,
-                chains_by_style=chains_by_style,
-                primary_exit_style=exit_style,
-            )
-            
-            if hasattr(data_src, 'close'):
-                data_src.close()
-            
-            if verbose:
-                print(f"  Completed run: {run.run_id}")
-            return run.run_id
-            
-        except Exception as e:
-            if 'run' in locals():
-                fail_run(db, run, str(e))
-            raise
+                    }
+                )
+
+        num_trades = int(metrics.get("num_trades", 0) or 0)
+        net_profit_usd = float(metrics.get("final_pnl", 0.0) or 0.0)
+        gross_profit = float(sum(float(t.pnl) for t in trades if float(t.pnl) > 0.0))
+        gross_loss = float(abs(sum(float(t.pnl) for t in trades if float(t.pnl) < 0.0)))
+        if gross_loss > 0:
+            profit_factor = gross_profit / gross_loss
+        else:
+            profit_factor = 10.0 if gross_profit > 0 else 0.0
+        avg_trade_usd = (net_profit_usd / num_trades) if num_trades > 0 else 0.0
+        final_pnl_vs_hodl = float(sum(float(getattr(t, "pnl_vs_hodl", 0.0) or 0.0) for t in trades))
+
+        initial_equity = float(getattr(config, "initial_capital_usd", 0.0) or 0.0)
+        final_equity = initial_equity + net_profit_usd
+        net_profit_pct = (net_profit_usd / initial_equity * 100.0) if initial_equity > 0 else 0.0
+        max_dd_pct = float(metrics.get("max_drawdown_pct", 0.0) or 0.0)
+
+        formatted_metrics = {
+            "initial_equity": initial_equity,
+            "final_equity": final_equity,
+            "net_profit_usd": net_profit_usd,
+            "net_profit_pct": net_profit_pct,
+            "final_pnl_vs_hodl": final_pnl_vs_hodl,
+            "max_drawdown_pct": max_dd_pct,
+            "max_drawdown_usd": (max_dd_pct / 100.0) * initial_equity if initial_equity > 0 else 0.0,
+            "num_trades": num_trades,
+            "win_rate": float(metrics.get("win_rate", 0.0) or 0.0) * 100.0,
+            "profit_factor": float(profit_factor),
+            "avg_trade_usd": float(avg_trade_usd),
+            "sharpe_ratio": 0.0,
+            "sortino_ratio": 0.0,
+        }
+
+        run_result.status = "finished"
+        run_result.metrics = {exit_style: formatted_metrics}
+        run_result.recent_chains = {exit_style: chains_list}
+        save_run_result(run_result)
+        update_run_status(run_id, "finished")
+
+        if hasattr(data_src, "close"):
+            data_src.close()
+
+        if verbose:
+            print(f"  Completed run: {run_id}")
+        return run_id
+
+    except Exception as e:
+        update_run_status(run_id, "failed", error=str(e))
+        raise
 
 
 def run_synthetic_vs_live_pair(
@@ -222,10 +304,9 @@ def run_synthetic_vs_live_pair(
 
 def get_metrics_for_run(run_id: str, exit_style: str) -> Optional[Dict[str, Any]]:
     """Get metrics for a completed run."""
-    with get_db_session() as db:
-        result = get_run_with_details(db, run_id)
-        if not result:
-            return None
-        
-        metrics = result.get("metrics", {})
-        return metrics.get(exit_style, {})
+    from src.backtest.run_store import load_result
+
+    result = load_result(run_id)
+    if not result:
+        return None
+    return (result.metrics or {}).get(exit_style, {})
diff --git a/src/backtest/covered_call_simulator.py b/src/backtest/covered_call_simulator.py
index f1605cb..de84c58 100644
--- a/src/backtest/covered_call_simulator.py
+++ b/src/backtest/covered_call_simulator.py
@@ -23,6 +23,7 @@ from .pricing import bs_call_price, bs_call_delta, get_synthetic_iv, compute_rea
 from src.models import MarketContext
 from src.metrics.volatility import compute_ivrv_ratio
 from src.scoring.candidates import score_option_candidate
+from .units import assert_premium_usd_sane
 
 State = Dict[str, Any]
 PolicyFn = Callable[[State], bool]
@@ -442,6 +443,9 @@ class CoveredCallSimulator:
         if spot_df.empty:
             return None
 
+        spot_at_open = float(spot_df["close"].iloc[0])
+        assert_premium_usd_sane(open_price, spot_at_open, context=f"simulate_single_call {target.instrument_name}")
+
         opt_df = ds.get_option_ohlc(
             instrument_name=target.instrument_name,
             start=decision_time,
@@ -703,6 +707,9 @@ class CoveredCallSimulator:
             open_price = float(option_snapshot.mark_price or 0.0)
             if open_price <= 0:
                 return None
+
+            spot_at_open = float(spot_df["close"].iloc[0])
+            assert_premium_usd_sane(open_price, spot_at_open, context=f"open {instrument_name}")
             
             opt_df = ds.get_option_ohlc(
                 instrument_name=instrument_name,
diff --git a/src/backtest/diff.py b/src/backtest/diff.py
index 98d9091..ec524f6 100644
--- a/src/backtest/diff.py
+++ b/src/backtest/diff.py
@@ -1,11 +1,12 @@
-"""
-Reusable diff logic for comparing backtest runs.
+"""Reusable diff logic for comparing backtest runs.
+
+This module is intentionally DB-free and reads backtest metrics from
+src/backtest/run_store.py, matching the Backtest Lab's file-based storage.
 """
 
-from typing import Dict, Any, Optional, List, Tuple
+from typing import Any, Dict, List, Optional, Tuple
 
-from src.db import get_db_session
-from src.db.models_backtest import BacktestRun, BacktestMetric
+from src.backtest.run_store import load_result
 
 
 METRICS_FIELDS: List[Tuple[str, str]] = [
@@ -23,27 +24,15 @@ METRICS_FIELDS: List[Tuple[str, str]] = [
 ]
 
 
-def fetch_run(db, run_id: str) -> Optional[BacktestRun]:
-    """Fetch a backtest run by its run_id string."""
-    return db.query(BacktestRun).filter(BacktestRun.run_id == run_id).first()
-
-
-def fetch_metrics(db, run_numeric_id: int, exit_style: str) -> Optional[BacktestMetric]:
-    """Fetch metrics for a run by numeric ID and exit style."""
-    return db.query(BacktestMetric).filter(
-        BacktestMetric.run_id == run_numeric_id,
-        BacktestMetric.exit_style == exit_style,
-    ).first()
-
-
-def get_metric_value(metrics: BacktestMetric, field: str) -> float:
-    """Extract a metric value, returning 0.0 if not found."""
-    if hasattr(metrics, field):
-        val = getattr(metrics, field)
-        return float(val) if val is not None else 0.0
-    if hasattr(metrics, 'metrics_json') and metrics.metrics_json:
-        return float(metrics.metrics_json.get(field, 0.0))
-    return 0.0
+def get_metric_value(metrics: Dict[str, Any], field: str) -> float:
+    """Extract a metric value from a run_store metrics dict."""
+    try:
+        val = metrics.get(field)
+        if val is None:
+            return 0.0
+        return float(val)
+    except Exception:
+        return 0.0
 
 
 def format_value(val: float, fmt_type: str) -> str:
@@ -99,80 +88,60 @@ def compute_diff_for_runs(
     Raises:
         ValueError if runs or metrics not found
     """
-    with get_db_session() as db:
-        run_a = fetch_run(db, run_id_a)
-        if not run_a:
-            raise ValueError(f"Run A not found: {run_id_a}")
-        
-        run_b = fetch_run(db, run_id_b)
-        if not run_b:
-            raise ValueError(f"Run B not found: {run_id_b}")
-        
-        if exit_style:
-            effective_exit_style = exit_style
-        else:
-            exit_style_a = run_a.primary_exit_style
-            exit_style_b = run_b.primary_exit_style
-            
-            if exit_style_a != exit_style_b:
-                raise ValueError(
-                    f"Runs have different primary exit styles "
-                    f"(A={exit_style_a}, B={exit_style_b}). "
-                    f"Please specify exit_style explicitly."
-                )
-            effective_exit_style = exit_style_a
-        
-        metrics_a = fetch_metrics(db, run_a.id, effective_exit_style)
-        if not metrics_a:
-            raise ValueError(
-                f"Metrics not found for run A ({run_id_a}) "
-                f"with exit_style={effective_exit_style}"
-            )
-        
-        metrics_b = fetch_metrics(db, run_b.id, effective_exit_style)
-        if not metrics_b:
-            raise ValueError(
-                f"Metrics not found for run B ({run_id_b}) "
-                f"with exit_style={effective_exit_style}"
-            )
-        
-        run_a_metadata = {
-            "run_id": run_a.run_id,
-            "underlying": run_a.underlying,
-            "data_source": run_a.data_source,
-            "start_ts": run_a.start_ts.isoformat() if run_a.start_ts else None,
-            "end_ts": run_a.end_ts.isoformat() if run_a.end_ts else None,
-            "decision_interval_minutes": run_a.decision_interval_minutes,
-        }
-        
-        run_b_metadata = {
-            "run_id": run_b.run_id,
-            "underlying": run_b.underlying,
-            "data_source": run_b.data_source,
-            "start_ts": run_b.start_ts.isoformat() if run_b.start_ts else None,
-            "end_ts": run_b.end_ts.isoformat() if run_b.end_ts else None,
-            "decision_interval_minutes": run_b.decision_interval_minutes,
-        }
-        
-        metrics_dict = {}
-        for field, fmt_type in METRICS_FIELDS:
-            val_a = get_metric_value(metrics_a, field)
-            val_b = get_metric_value(metrics_b, field)
-            diff = val_b - val_a
-            
-            metrics_dict[field] = {
-                "a": val_a,
-                "b": val_b,
-                "diff": diff,
-                "fmt_type": fmt_type,
-            }
-        
-        return {
-            "run_a": run_a_metadata,
-            "run_b": run_b_metadata,
-            "exit_style": effective_exit_style,
-            "metrics": metrics_dict,
-        }
+    run_a = load_result(run_id_a)
+    if not run_a:
+        raise ValueError(f"Run A not found: {run_id_a}")
+
+    run_b = load_result(run_id_b)
+    if not run_b:
+        raise ValueError(f"Run B not found: {run_id_b}")
+
+    if exit_style:
+        effective_exit_style = exit_style
+    else:
+        effective_exit_style = run_a.config.get("exit_style") or run_b.config.get("exit_style")
+        if not effective_exit_style:
+            raise ValueError("exit_style not provided and not present in run configs")
+
+    metrics_a = (run_a.metrics or {}).get(effective_exit_style)
+    if not metrics_a:
+        raise ValueError(f"Metrics not found for run A ({run_id_a}) with exit_style={effective_exit_style}")
+
+    metrics_b = (run_b.metrics or {}).get(effective_exit_style)
+    if not metrics_b:
+        raise ValueError(f"Metrics not found for run B ({run_id_b}) with exit_style={effective_exit_style}")
+
+    run_a_metadata = {
+        "run_id": run_a.run_id,
+        "underlying": run_a.config.get("underlying"),
+        "data_source": run_a.config.get("data_source"),
+        "start_ts": run_a.config.get("start_date") or run_a.config.get("start"),
+        "end_ts": run_a.config.get("end_date") or run_a.config.get("end"),
+        "decision_interval_minutes": run_a.config.get("decision_interval_minutes"),
+    }
+
+    run_b_metadata = {
+        "run_id": run_b.run_id,
+        "underlying": run_b.config.get("underlying"),
+        "data_source": run_b.config.get("data_source"),
+        "start_ts": run_b.config.get("start_date") or run_b.config.get("start"),
+        "end_ts": run_b.config.get("end_date") or run_b.config.get("end"),
+        "decision_interval_minutes": run_b.config.get("decision_interval_minutes"),
+    }
+
+    metrics_dict: Dict[str, Any] = {}
+    for field, fmt_type in METRICS_FIELDS:
+        val_a = get_metric_value(metrics_a, field)
+        val_b = get_metric_value(metrics_b, field)
+        d = val_b - val_a
+        metrics_dict[field] = {"a": val_a, "b": val_b, "diff": d, "fmt_type": fmt_type}
+
+    return {
+        "run_a": run_a_metadata,
+        "run_b": run_b_metadata,
+        "exit_style": effective_exit_style,
+        "metrics": metrics_dict,
+    }
 
 
 def print_diff_report_from_data(diff_data: Dict[str, Any]) -> None:
diff --git a/src/backtest/fidelity_suite.py b/src/backtest/fidelity_suite.py
new file mode 100644
index 0000000..e8e5863
--- /dev/null
+++ b/src/backtest/fidelity_suite.py
@@ -0,0 +1,239 @@
+"""Lab-based Fidelity Suite Orchestrator.
+
+TOP CONSTRAINT (TOC): trustworthiness.
+This orchestrator intentionally runs Fidelity through the *existing Backtest Lab*
+compare + diff utilities so the score reflects the same machinery users rely on.
+
+STEP 0 — DISCOVERY NOTES (MANDATORY)
+- Backtest lab compare runner:
+  - src/backtest/compare.py
+  - function: run_synthetic_vs_live_pair(...)
+- Backtest lab diff utilities:
+  - src/backtest/diff.py
+  - function: compute_diff_for_runs(run_id_a, run_id_b, exit_style=...)
+- Backtest lab run store conventions (pattern reference):
+  - src/backtest/run_store.py
+  - env override pattern, index.jsonl append, latest pointer
+- FastAPI route registration:
+  - src/web/routes_fidelity.py defines /api/fidelity/latest and /api/fidelity/history
+  - src/web_app.py registers routers via app.include_router(...)
+- Backtest Lab UI rendering:
+  - src/web/dashboard.py (render_dashboard_html)
+
+This module provides a deterministic, robust MVP scoring layer on top of Lab diff.
+"""
+
+from __future__ import annotations
+
+import math
+import uuid
+from dataclasses import dataclass
+from datetime import datetime, timezone
+from typing import Any, Dict, List, Optional, Tuple
+
+from src.backtest import compare, diff
+
+
+METRIC_TOLERANCES: Dict[str, float] = {
+    "net_profit_pct": 5.0,  # percentage points
+    "max_drawdown_pct": 5.0,  # percentage points
+    "win_rate": 10.0,  # percentage points
+    "profit_factor": 0.30,  # absolute
+    "avg_trade_usd": 25.0,  # USD
+}
+
+
+def _metric_score(diff_value: float, tolerance: float) -> float:
+    if tolerance <= 0:
+        return 0.0
+    err_ratio = abs(float(diff_value)) / float(tolerance)
+    return float(100.0 * math.exp(-1.2 * err_ratio))
+
+
+def score_case_from_diff(diff_payload: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
+    """Compute a case score from a diff payload.
+
+    Returns: (case_score, diagnostics)
+    """
+    metrics = (diff_payload or {}).get("metrics") or {}
+
+    used: Dict[str, float] = {}
+    skipped: List[str] = []
+    scores: List[float] = []
+
+    for field, tol in METRIC_TOLERANCES.items():
+        m = metrics.get(field)
+        if not isinstance(m, dict) or "diff" not in m:
+            skipped.append(field)
+            continue
+        used[field] = tol
+        scores.append(_metric_score(m.get("diff", 0.0), tol))
+
+    if not scores:
+        return 0.0, {"used_metrics": used, "skipped_metrics": skipped}
+
+    return float(sum(scores) / len(scores)), {"used_metrics": used, "skipped_metrics": skipped}
+
+
+def _safe_get_num_trades(diff_payload: Dict[str, Any]) -> Tuple[Optional[int], Optional[int]]:
+    try:
+        metrics = (diff_payload or {}).get("metrics") or {}
+        mt = metrics.get("num_trades") or {}
+        # In diff.py: a = run_a (typically synthetic), b = run_b (typically live)
+        a = mt.get("a")
+        b = mt.get("b")
+        if a is None or b is None:
+            return None, None
+        return int(a), int(b)
+    except Exception:
+        return None, None
+
+
+@dataclass(frozen=True)
+class FidelityCase:
+    exit_style: str
+    synth_run_id: Optional[str]
+    live_run_id: Optional[str]
+    diff_payload: Optional[Dict[str, Any]]
+    num_trades_synth: Optional[int]
+    num_trades_live: Optional[int]
+    case_score: float
+    valid: bool
+    error: Optional[str] = None
+
+    def to_dict(self) -> Dict[str, Any]:
+        payload: Dict[str, Any] = {
+            "exit_style": self.exit_style,
+            "synth_run_id": self.synth_run_id,
+            "live_run_id": self.live_run_id,
+            "diff": self.diff_payload,
+            "num_trades": {"synth": self.num_trades_synth, "live": self.num_trades_live},
+            "case_score": self.case_score,
+            "valid": self.valid,
+        }
+        if self.error:
+            payload["error"] = self.error
+        return payload
+
+
+def run_fidelity_from_lab(
+    underlying: str,
+    start_ts: datetime,
+    end_ts: datetime,
+    decision_interval_minutes: int = 60,
+    exit_styles: List[str] | None = None,
+    min_trades_per_case: int = 5,
+) -> Dict[str, Any]:
+    """Run Lab-based fidelity across a set of exit styles.
+
+    Runs synthetic vs live via src/backtest/compare.py and compares via src/backtest/diff.py.
+
+    Returns a report dict suitable for persistence via src/backtest/fidelity_store.py.
+    """
+
+    if exit_styles is None:
+        exit_styles = ["hold_to_expiry", "tp_and_roll"]
+
+    created_at = datetime.now(timezone.utc).isoformat()
+    run_id = f"fidelity_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{underlying}_{uuid.uuid4().hex[:8]}"
+
+    cases: List[FidelityCase] = []
+
+    for exit_style in exit_styles:
+        synth_run_id: Optional[str] = None
+        live_run_id: Optional[str] = None
+        diff_payload: Optional[Dict[str, Any]] = None
+        err: Optional[str] = None
+        case_score = 0.0
+        num_trades_synth: Optional[int] = None
+        num_trades_live: Optional[int] = None
+        valid = False
+
+        try:
+            synth_run_id, live_run_id = compare.run_synthetic_vs_live_pair(
+                underlying=underlying,
+                start_ts=start_ts,
+                end_ts=end_ts,
+                decision_interval_minutes=decision_interval_minutes,
+                exit_style=exit_style,
+                verbose=False,
+            )
+
+            diff_payload = diff.compute_diff_for_runs(
+                run_id_a=synth_run_id,
+                run_id_b=live_run_id,
+                exit_style=exit_style,
+            )
+
+            num_trades_synth, num_trades_live = _safe_get_num_trades(diff_payload)
+
+            case_score, _diag = score_case_from_diff(diff_payload)
+

TRUNCATED
