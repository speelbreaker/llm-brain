# Recent Diff

generated_at_utc: 2025-12-22T17:02:24Z
branch: main
head_sha: f6e339582a45a38469c24cc6b14a910fdfc49bdd
base: origin/main

## git log --oneline -n 25
f6e3395 chore: refresh latest artifacts
fa5a78a Auto-commit 2025-12-22T16:38:19Z
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

## git diff --stat origin/main..HEAD
 .vscode/tasks.json                                 |    2 +-
 HEALTHCHECK.md                                     |  126 +-
 Makefile                                           |   14 +
 POEM.md                                            |   22 +
 README.md                                          |   41 +
 ROADMAP_BACKLOG.md                                 |   45 +
 agent_loop.py                                      |   21 +
 ...uct-minded-engineer-Update-th_1766256235012.txt |   73 -
 data/backtests/index.jsonl                         |   18 +
 .../history/20251219_234717/fidelity_report.json   |  131 -
 .../BTC/history/20251219_234717/fidelity_report.md |  147 -
 data/fidelity_runs/BTC/latest/fidelity_report.json |  131 -
 data/fidelity_runs/BTC/latest/fidelity_report.md   |  147 -
 docs/CONTEXT_PACK.md                               |  219 +-
 docs/FIDELITY_BTC_latest.json                      |  349 ++
 docs/FIDELITY_BTC_latest.md                        |  252 ++
 docs/OPS_HEALTH_latest.json                        |   10 +
 docs/RECENT_DIFF.md                                | 2002 ++++++++++
 docs/REPO_MANIFEST.json                            | 3903 ++++++++++++++++++++
 docs/ROADMAP_BACKLOG_latest.md                     |  858 +++++
 docs/TEST_SUMMARY_latest.txt                       |    2 +
 scripts/capture_pytest_summary.sh                  |   49 +
 scripts/gen_fidelity_latest_docs.py                |   94 +
 scripts/gen_ops_health_latest.py                   |  114 +
 scripts/gen_recent_diff.sh                         |   77 +
 scripts/gen_repo_manifest.py                       |  188 +
 scripts/print_fidelity_summary.py                  |   94 +
 scripts/push_context_pack_to_drive.sh              |   83 +
 scripts/roadmap_append_changelog.py                |  151 +
 scripts/run_fidelity_from_lab.py                   |   56 +
 scripts/run_fidelity_from_lab_daily.py             |   56 +
 scripts/sabotage_fidelity_drill.py                 |   42 +
 src/backtest/compare.py                            |  337 +-
 src/backtest/covered_call_simulator.py             |    7 +
 src/backtest/diff.py                               |  169 +-
 src/backtest/fidelity_store.py                     |  208 +-
 src/backtest/fidelity_suite.py                     |  404 ++
 src/backtest/live_deribit_data_source.py           |   29 +-
 src/backtest/manager.py                            |  147 +-
 src/backtest/pricing.py                            |   12 +-
 src/backtest/state_builder.py                      |   32 +-
 src/backtest/units.py                              |   58 +
 src/calibration_config.py                          |    1 +
 src/calibration_extended.py                        |    3 +
 src/calibration_update_policy.py                   |   82 +-
 src/config.py                                      |   30 +
 src/data/live_deribit_exam.py                      |   37 +
 src/db/__init__.py                                 |    1 +
 src/db/models_telegram.py                          |   21 +
 src/fidelity/canonical_strategies.py               |  194 +-
 src/fidelity/gating.py                             |   17 +
 src/fidelity/market_replay.py                      |  209 +-
 src/fidelity/ops_runner.py                         |  284 ++
 src/fidelity/reporting.py                          |    7 +
 src/fidelity/run_suite.py                          |  114 +-
 src/fidelity/scoring.py                            |   44 +-
 src/harvest_status.py                              |  271 ++
 src/healthcheck.py                                 |  364 +-
 src/ops/calibration_status.py                      |   98 +
 src/ops/facts_resolver.py                          |   89 +
 src/ops/fidelity_status.py                         |   91 +-
 src/ops/gate_factories.py                          |  442 +++
 src/ops/gates.py                                   |  112 +
 src/telegram/__init__.py                           |    1 +
 src/telegram/store.py                              |  108 +
 src/web/api_errors.py                              |   41 +
 src/web/dashboard.py                               |  313 +-
 src/web/routes_backtest.py                         |  131 +-
 src/web/routes_fidelity.py                         |   46 +
 src/web/routes_health.py                           |   37 +-
 src/web/routes_positions.py                        |   12 +
 src/web/routes_telegram.py                         |  214 ++
 src/web_app.py                                     |   18 +
 tests/test_api_calibration_run_with_policy.py      |   54 +
 tests/test_api_fidelity_endpoints.py               |   64 +
 tests/test_backtest_greg_modes.py                  |   48 +-
 tests/test_backtest_preflight.py                   |  214 ++
 tests/test_calibration_update_policy.py            |    2 +
 tests/test_context_pack.py                         |   66 +
 tests/test_fidelity_canonical_store.py             |  125 +
 tests/test_fidelity_gate_integration.py            |   84 +
 tests/test_fidelity_lab_scoring.py                 |   25 +
 tests/test_fidelity_latest_docs_generator.py       |   52 +
 tests/test_fidelity_latest_resolution.py           |   44 +-
 tests/test_fidelity_missing_close.py               |  126 +
 ..._fidelity_moneyness_fallback_and_diagnostics.py |   68 +
 tests/test_gen_ops_health_latest.py                |   61 +
 tests/test_health_and_calibration_automation.py    |   93 +-
 tests/test_healthcheck_basic.py                    |  259 +-
 tests/test_healthcheck_config.py                   |   15 +-
 tests/test_live_deribit_units.py                   |   53 +
 tests/test_ops_fidelity_clamping_and_schema.py     |  102 +
 tests/test_ops_fidelity_coverage_penalty.py        |   36 +
 tests/test_ops_health_artifact.py                  |   40 +
 tests/test_ops_health_endpoints.py                 |  809 ++++
 tests/web/expected_routes.json                     |  443 ++-
 tests/web/test_telegram_webhook.py                 |  126 +
 97 files changed, 15627 insertions(+), 1734 deletions(-)

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
index 93e6324..25e7299 100644
--- a/HEALTHCHECK.md
+++ b/HEALTHCHECK.md
@@ -4,7 +4,131 @@ This document lists quick commands to verify that core parts of the system are w
 
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
+- **Truth (facts)**: raw observations from the filesystem/stores (harvest presence + age, calibration last run, fidelity last run).
+- **Trust (gates)**: normalized gate results with explicit `mode` (`off|warn|block`) and `status` (`PASS|WARN|FAIL`).
+- **Trade (policy)**: aggregated `gate_overall` (`status|severity|can_trade`) used by dashboards and automation.
+- **Decisions**: `overall_status`/`summary` are derived from `gate_overall` whenever gates are available; `checks_overall`/`checks_summary` are kept purely for diagnostics.
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
+### Fidelity Report Store (Canonical)
+
+Ops-grade Synthetic Fidelity is persisted in a single canonical, file-based store:
+
+- Base dir: `data/fidelity_runs/` (override via `FIDELITY_DIR` or `FIDELITY_RUNS_DIR`)
+- Per-run report: `data/fidelity_runs/<run_id>.json`
+- Latest (full report): `data/fidelity_runs/latest.json`
+- Latest per underlying (full report): `data/fidelity_runs/BTC/latest.json`, `data/fidelity_runs/ETH/latest.json`
+
+The latest *summary* for the dashboard endpoints is maintained separately:
+
+- `data/fidelity_runs/latest_summary.json`
+- `data/fidelity_runs/index.jsonl`
+
+**Schema highlights** (fields health/gates rely on):
+
+- `run_id`, `created_at`, `underlying`
+- `component_scores`: `underlying_returns`, `iv_surface_level`, `spot_iv_coupling`, `strategy_pnl_parity`
+- `overall_score` (weighted combination), `gate_label` (`TRUSTED|WARNING|UNTRUSTED`)
+- `thresholds`: `trusted_threshold`, `warn_threshold`, `min_coverage_ratio`
+- `coverage.strategy_pnl_parity`: `valid_cases`, `total_cases`, `coverage_ratio_cases`, `min_trades_per_case`
+
+### Running the Ops-Grade Fidelity Suite
+
+This produces a canonical report in `data/fidelity_runs/` that is consumed by ops health and the unified gates.
+
+```bash
+python -c "from src.fidelity.ops_runner import run_ops_fidelity_suite; run_ops_fidelity_suite(underlying='BTC', start_ts=1735689600, end_ts=1736121600)"
+```
+
+> Strategy PnL parity uses Backtest Lab paired runs (`compare.run_synthetic_vs_live_pair`) and diffs (`diff.compute_diff_for_runs`).
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
 
diff --git a/Makefile b/Makefile
new file mode 100644
index 0000000..07d1860
--- /dev/null
+++ b/Makefile
@@ -0,0 +1,14 @@
+context-pack:
+	python3 scripts/gen_repo_manifest.py
+	bash scripts/gen_recent_diff.sh
+	python3 scripts/gen_fidelity_latest_docs.py
+
+extras:
+	python3 scripts/gen_ops_health_latest.py
+
+context-pack-push: context-pack extras
+	@if [ ! -f docs/TEST_SUMMARY_latest.txt ]; then \
+		printf "%s\n%s\n" "$$(date -u +%Y-%m-%dT%H:%MZ)" "pytest summary unavailable" > docs/TEST_SUMMARY_latest.txt; \
+	fi
+	cp ROADMAP_BACKLOG.md docs/ROADMAP_BACKLOG_latest.md
+	@echo "Updated docs/ROADMAP_BACKLOG_latest.md (upload handled externally)"
diff --git a/POEM.md b/POEM.md
new file mode 100644
index 0000000..2744d6e
--- /dev/null
+++ b/POEM.md
@@ -0,0 +1,22 @@
+# The Algorithm's Dance
+
+In circuits deep where logic flows,
+A trader wakes, the market knows.
+Each five-minute tick, a chance to see
+What volatility might decree.
+
+The Greeks align in perfect form,
+Delta, theta, ride the storm.
+Black-Scholes whispers in the code,
+Where synthetic prices find their road.
+
+Rule-based mind or LLM's grace,
+Both paths converge at risk's gate.
+Backtests replay what might have been,
+The equity curve tells its tale.
+
+In testnet's safe and sandboxed realm,
+The agent learns, the agent grows.
+So here's to code that trades with care,
+A faithful servant, always there.
+
diff --git a/README.md b/README.md
index dea428a..0752cb6 100644
--- a/README.md
+++ b/README.md
@@ -92,6 +92,47 @@ python agent_loop.py
 python -m backtest.env_simulator
 ```
 
+## Context Pack & Drive Publishing
+
+This repo can generate a deterministic "context pack" under `docs/` for external consumers (e.g., a Google Drive folder used as LLM context).
+
+### Generate latest artifacts locally
+
+```bash
+# Generates docs/REPO_MANIFEST.json, docs/RECENT_DIFF.md, and other "latest" artifacts
+make context-pack-push
+```
+
+### Fidelity “latest” artifacts
+
+If Fidelity has been run and the canonical store exists, the context-pack generator will also publish Fidelity reports into `docs/`:
+
+- `docs/FIDELITY_BTC_latest.json` / `docs/FIDELITY_BTC_latest.md`
+  - copied from `data/fidelity_runs/BTC/latest/fidelity_report.json` and `.md`
+- `docs/FIDELITY_ETH_latest.json` / `docs/FIDELITY_ETH_latest.md` (if present)
+  - copied from `data/fidelity_runs/ETH/latest/fidelity_report.json` and `.md`
+
+Missing sources do **not** fail context-pack generation; the generator prints a warning and skips those files.
+
+You can also run the generator directly:
+
+```bash
+python3 scripts/gen_fidelity_latest_docs.py
+```
+
+### Upload to Google Drive (rclone)
+
+If you use `rclone` to publish the `docs/` “latest” artifacts to Drive, use:
+
+```bash
+bash scripts/push_context_pack_to_drive.sh
+```
+
+Notes:
+- Requires `rclone` configured with a `gdrive:` remote.
+- The target can be overridden with `CONTEXT_PACK_DRIVE_REMOTE`.
+- To also upload a timestamped snapshot under `history/`, set `CONTEXT_PACK_UPLOAD_HISTORY=1`.
+
 ## Project Structure
 
 ```
diff --git a/ROADMAP_BACKLOG.md b/ROADMAP_BACKLOG.md
index 21bce55..131a69a 100644
--- a/ROADMAP_BACKLOG.md
+++ b/ROADMAP_BACKLOG.md
@@ -811,3 +811,48 @@ The system supports three levels of automation, rolled out incrementally:
 5. **Audit trail:** Every run, every LLM suggestion, and every parameter change is logged and traceable.
 
 ---
+
+## Changelog (auto)
+- (entries appended newest-first)
+
+- 2025-12-22T16:55Z [COPILOT] sha=fa5a78a
+  - Summary: Refresh TEST_SUMMARY_latest.txt after full-suite run; Refresh context-pack latest artifacts prior to push
+  - Tests: 822 passed, 5 skipped, 53 warnings in 335.18s (0:05:35)
+  - Endpoints: none
+  - Context-pack: uploaded (no)
+- 2025-12-21T21:51Z [COPILOT] sha=3bc7e86
+  - Summary: Document context-pack + Drive publishing (rclone) in README; Document Fidelity latest artifacts in docs/
+  - Tests: not run (docs-only)
+  - Endpoints: none
+  - Context-pack: uploaded (no)
+- 2025-12-21T20:38Z [CODEx] sha=3bc7e86546adf507d277ca9f65342246a7adc804
+  - Summary: Gate_overall/summary drive decisions while checks stay diagnostic; - Fidelity gate check no longer toggles can_trade; - Harvest requirement + ops dashboard/tests align with truth→trust pipeline
+  - Tests: ======================= 34 passed, 10 warnings in 7.82s ========================
+  - Endpoints: /api/ops/health/run, /api/ops/health/status
+  - Context-pack: uploaded (yes)
+- 2025-12-21T20:34Z [COPILOT] sha=3bc7e86
+  - Summary: Fix ROADMAP_BACKLOG duplicate Changelog (auto) header; Update changelog append script to target last section
+  - Tests: not run (docs/script-only change)
+  - Endpoints: none
+  - Context-pack: uploaded (no)
+- 2025-12-21T20:31Z [COPILOT] sha=3bc7e86
+  - Summary: Add OPS_HEALTH_latest.json context-pack artifact generator (fake mode for tests); Wire ops health + test summary into context-pack-push; Make ops fidelity auditable: raw scores + clamping warnings + schema hard-fail
+  - Tests: 5 passed, 4 warnings in 10.55s
+  - Endpoints: none
+  - Context-pack: uploaded (no)
+- 2025-12-21T20:04Z [COPILOT] sha=3bc7e86
+  - Summary: Make ops fidelity conservative: clamp scores + apply coverage penalty; Remove misleading gate_label parameters; missing_close always UNTRUSTED; Add tests for penalty + coverage schema
+  - Tests: 28 passed, 5 warnings in 73.80s (0:01:13)
+  - Endpoints: none
+  - Context-pack: uploaded (no)
+- 2025-12-21T19:22Z [CODEx] sha=3bc7e86546adf507d277ca9f65342246a7adc804
+  - Summary: Gate-overall authoritative status/summary + checks_overall diagnostics; - Fidelity gate mode driven by env only; - Harvest required toggles + ops tests/UI updates
+  - Tests: ======================= 31 passed, 10 warnings in 5.48s ========================
+  - Endpoints: /api/ops/health/run, /api/ops/health/status
+  - Context-pack: uploaded (yes)
+
+- 2025-12-21T19:10Z [COPILOT] sha=3bc7e86
+  - Summary: Add roadmap changelog automation helper; Add pytest summary capture script (+ optional changelog append); Add context-pack-push target to refresh ROADMAP_BACKLOG_latest
+  - Tests: 6 passed, 4 warnings in 1.50s
+  - Endpoints: none
+  - Context-pack: uploaded (no)
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
diff --git a/docs/CONTEXT_PACK.md b/docs/CONTEXT_PACK.md
index b05b534..1cbf497 100644
--- a/docs/CONTEXT_PACK.md
+++ b/docs/CONTEXT_PACK.md
@@ -1,210 +1,29 @@
-# Options Trading Agent — Context Pack (Canonical)
+# Repo Context Pack
 
-**Purpose:** Single source of truth for humans + AI Builder about what exists, how it’s wired, and what “done” means.  
-**Rule:** Prefer *facts about the repo* over opinions. If something is not verified in code/config, label it **Planned**.
+This repo ships two generated artifacts to help LLMs stay repo-aware without bundling zips.
 
-**Last updated:** 2025-12-16
+## Outputs
 
----
+- `docs/REPO_MANIFEST.json`: repo tree, git metadata, hotspots, endpoints index, and important paths.
+- `docs/RECENT_DIFF.md`: recent git history and diff from a base ref.
 
-## 0) How to Use This File
+## Usage
 
-When you (or an AI Builder) are implementing changes:
+Generate both artifacts:
 
-1) **Repo-first:** read the relevant files before proposing edits.  
-2) **Keep invariants sacred:** risk rails are not “nice to have.”  
-3) **No “Done” claims without evidence:** if you claim something is implemented, reference a file path + behavior.
+```bash
+make context-pack
+```
 
----
+Or run each generator directly:
 
-## 1) What the App Does Today (Current Reality)
+```bash
+python3 scripts/gen_repo_manifest.py
+bash scripts/gen_recent_diff.sh
+```
 
-- Automates **BTC/ETH covered-call trading** on Deribit **testnet** with **two decision paths**:
-  - **Rule-based policy** (deterministic)
-  - **LLM co-pilot** that returns structured JSON actions
-- Runs a continuous **agent loop** that:
-  - builds market/portfolio state
-  - selects an action
-  - enforces risk limits
-  - executes (dry-run by default unless enabled)
-  - logs outcomes for audit/training
-- Provides a **FastAPI web dashboard** (tabs typically include Live Agent, Backtesting Lab, Calibration, and Chat/Explain).
-- Includes a **backtesting engine**, synthetic pricing tools, and training data exporters (for RL/ML experiments).
-
----
-
-## 2) Architecture Map (Modules & Key Flows)
-
-### Entry Points
-- `agent_loop.py` — orchestrates the live loop
-- `src/web_app.py` — serves the dashboard (and may boot/host the agent loop)
-
-### Core Wiring (high level)
-- **Config:** `src/config.py` (Pydantic settings)
-- **State construction:**
-  - Live: `src/state_builder.py` + shared `src/state_core.py` (`AgentState`)
-  - Backtests: `src/backtest/state_builder.py`
-- **Decision (“brains”):**
-  - Rule-based: `src/policy_rule_based.py`
-  - LLM brain: `src/agent_brain_llm.py` (OpenAI decision JSON)
-- **Strategy layer:**
-  - Strategy interface + current covered-call implementation: `src/strategies/covered_call.py`
-  - Registry: `src/strategies/registry.py`
-- **Risk & execution:**
-  - `src/risk_engine.py` enforces invariants
-  - `src/execution.py` places/simulates Deribit orders
-- **Market data:**
-  - Live authenticated: `src/deribit_client.py`
-  - Backtest/public: `src/backtest/deribit_client.py`
-  - Context aggregation: `src/market_context.py`
-
-### Backtest Stack (covered calls)
-- `src/backtest/covered_call_simulator.py`
-- `src/backtest/pricing.py`
-- `src/backtest/manager.py`
-- `src/backtest/config_presets.py`
-- `src/backtest/types.py`
-
-### Training / Utilities / Ops
-- `src/training_policy.py`, `src/training_profiles.py`
-- `src/calibration.py`
-- `src/synthetic_skew.py`
-- `src/chat_with_agent.py`
-- `scripts/*` helpers
-
-### Data + Logs
-- Decision logs: `logs/` (JSONL append-only)
-- Datasets/model artifacts: `data/`
-
-> **Planned (preferred direction):** modularize `src/web_app.py` into a `src/web/` package with routers + dashboard templates (see backlog).
-
----
-
-## 3) Strategy Catalog
-
-### 3.1 CoveredCallStrategy (current default)
-- A “wrapper strategy” that can run **rule-based** or **LLM** mode depending on settings.
-- Training profiles exist (conservative/moderate/aggressive) in `src/training_profiles.py`.
-
-### 3.2 GregBot (Magadini VRP Harvester) — **Bundle, not a single strategy**
-**Key point:** “GregBot” should be treated as a **selector + management framework** that can fire *different strategies* based on sensors.  
-This is exactly why your UI/backtest results must record **which sub-strategy fired, when, and P&L attribution**.
-
-**Example strategy codes (bundle):**
-- `STRATEGY_A_STRADDLE` — ATM straddle (VRP + regime dependent)
-- `STRATEGY_A_STRANGLE` — OTM strangle
-- `STRATEGY_B_CALENDAR` — calendar spread (term structure signal)
-- `STRATEGY_C_SHORT_PUT` — accumulation / short put (bullish + VRP)
-- `STRATEGY_D_IRON_BUTTERFLY` — defined risk for extreme IV rank
-- `STRATEGY_F_BULL_PUT_SPREAD` — defined risk bullish
-- `STRATEGY_F_BEAR_CALL_SPREAD` — defined risk bearish
-- `NO_TRADE` — safety filter triggered / no valid setup
-
-**Key sensors (must be computed + logged):**
-- `vrp_30d`, `vrp_7d` (Implied − Realized volatility premium)
-- `chop_factor_7d` (range-bound vs trending proxy)
-- `adx_14d` (trend strength)
-- `iv_rank_6m` (IV percentile)
-- `skew_25d` (put/call skew)
-- `term_structure_spread` (tenor IV spread)
-
-> **Important:** Numeric thresholds (e.g., VRP > 15) are *policy knobs*, not guaranteed truth. If thresholds exist, they must live in settings/config and be visible in the UI.
-
-### 3.3 Planned plug-ins (architecture supports; not necessarily implemented)
-Wheel, CrashHedge, and Spread strategies added via the `Strategy` interface + registry.
-
----
-
-## 4) Data Sources & Modes (Synthetic vs Live)
-
-### Primary Sources
-- **Live trading data (testnet):** via `src/deribit_client.py` (authenticated)  
-  balances, positions, option chains, execution, etc.
-- **Historical / public for backtests:** via `src/backtest/deribit_client.py` (mainnet candles + public option data when available)
-- **Synthetic pricing & smiles:** Black–Scholes in `src/backtest/pricing.py` + skew/regime helpers like `src/synthetic_skew.py`
-- **Calibration:** `src/calibration.py` + scripts like `scripts/compare_synthetic_vs_live.py` to compare synthetic vs live IV/smiles
-
-### Backtest / Scan modes (conceptual contract)
-- `BacktestType.GENERIC` — full P&L simulation (positions, rolls, exits)
-- `BacktestType.GREG_SELECTOR` — selector-only analysis (which sub-strategy would fire; fast)
-
-### Selector Data Sources (contract the UI should expose)
-- `SelectorDataSource.SYNTHETIC` — synthetic universe
-- `SelectorDataSource.HARVESTER` — historical harvested snapshots (if present)
-- `SelectorDataSource.LIVE` — current live snapshot
-
-### IV sourcing modes (contract the selector should support)
-- `iv_mode="synthetic"` — synthetic IV surface
-- `iv_mode="live"` — Deribit live IV
-- `iv_mode="hybrid"` — live IV with synthetic fallback
-
----
-
-## 5) Risk Invariants (Hard Rules)
-
-These are “stop-the-line” constraints. If violated, the system must choose `DO_NOTHING` and log why.
-
-- **Kill switch:** global halt blocks all non-`DO_NOTHING` actions
-- **Portfolio validity:** no trading when equity is missing/zero (guards missing private API / unfunded accounts)
-- **Margin discipline:** block at/above `max_margin_used_pct`; warn before limit (e.g., 90% of max)
-- **Delta guardrail:** `abs(net_delta) <= max_net_delta_abs`
-- **Daily drawdown cap:** optional daily stop if breached (UTC day)
-- **Per-expiry exposure:** projected short call size per expiry must remain covered + under `max_expiry_exposure`
-- **Liquidity gates:** min open interest + max spread requirements for candidates
-- **Training bypass:** if `is_training_on_testnet` is enabled, risk checks may be bypassed (must be loudly logged)
-
-> **Security requirements (policy):** never log secrets; explicit timeouts on external calls; webhook signature verification where used; return 503 when required secrets/config are missing.
-
----
-
-## 6) Observability & Artifacts
-
-Minimum viable observability:
-- Every decision log must include: **timestamp, market context, chosen action, rationale, risk pass/fail reasons, execution result**
-- For GregBot: log **sub-strategy code**, sensor values, and why alternatives were rejected
-- Keep **append-only JSONL** for traceability; DB can be added later, but don’t lose the recorder
-
----
-
-## 7) Current Backlog (Top Priorities)
-
-1) **Selector/UI config correctness:** when GregBot is selected, hide/disable generic exit knobs (hold-to-expiry/take-profit/roll) that don’t apply; show Greg-specific controls instead.  
-2) **GregBot attribution:** backtests must report which sub-strategy fired, when, count, and P&L per sub-strategy.  
-3) **Selector scan sources:** extend selector scans beyond synthetic to **LIVE** and **HARVESTER** where available, including `iv_mode` handling.  
-4) **Database-first position/decision store:** SQLite/Postgres with JSONL as recorder (dual-write or migration path).  
-5) **Watchdogs & process health:** heartbeat, rate-limit guards, safe-mode, kill-switch enforcement, restart policy.  
-6) **Position reconciliation & roll rules:** stronger tracker with audit trail; fewer silent mismatches.  
-7) **LLM safety rails:** strict schema validation, action whitelist, better fallbacks.  
-8) **Synthetic vs live realism:** tighten optimism factor, extend calibration, improve surface fit.  
-9) **Reduce coupling:** config injection, fewer globals, easier testing.  
-10) **Web app modularization:** split `src/web_app.py` into routers/templates to reduce “God module” coupling.
-
----
-
-## 8) Definition of Done (Non-Negotiable)
-
-A change is “done” only if all are true:
-
-- **Functionality:** matches the described behavior; strategy/registry wiring updated where relevant.
-- **Risk:** all actions still pass `risk_engine` invariants; kill-switch remains effective.
-- **UI:** dashboard endpoints/templates updated so the user can see and operate the feature (no “backend-only” delivery).
-- **Observability:** logs show action + rationale + risk reasons + execution result; errors are operator-readable.
-- **Docs:** update this Context Pack if behavior/interfaces/paths changed.
-
-### Tests (Mandatory Gates)
-- `python -m pytest -q` passes
-- **Every new endpoint ships with at least one endpoint-level test** (FastAPI `TestClient`)  
-  - include success path + at least one failure case (e.g., 400/401/403/404/422/503)
-- If an endpoint changes behavior/shape, update tests to prove it.
-
----
-
-## 9) Builder Prompt Standard (Required)
-
-Every Builder prompt must include:
-- **Read-first steps** (list tree, open key modules, summarize before editing)
-- **Scope + files to touch**
-- **Acceptance criteria** (including UI changes + logging)
-- **Tests required**
-- **Rollback plan** (how to revert safely)
+## Notes
 
+- `docs/RECENT_DIFF.md` redacts lines containing common secret env keys.
+- Large diffs are truncated after ~2000 lines and marked with `TRUNCATED`.
+- The diff base is `origin/main` when available, otherwise `HEAD~10`.
diff --git a/docs/FIDELITY_BTC_latest.json b/docs/FIDELITY_BTC_latest.json
new file mode 100644
index 0000000..ff44c7f
--- /dev/null
+++ b/docs/FIDELITY_BTC_latest.json
@@ -0,0 +1,349 @@
+{
+  "component_scores": {
+    "iv_surface_level": 100.0,
+    "spot_iv_coupling": 100.0,
+    "strategy_pnl_parity": 100.00000000000001,
+    "underlying_returns": 100.0
+  },
+  "component_status": {
+    "iv_surface_level": "not_available",
+    "spot_iv_coupling": "not_available",
+    "strategy_pnl_parity": "not_available",
+    "underlying_returns": "not_available"
+  },
+  "components": {
+    "iv_surface_level": {
+      "meta": {
+        "coverage": 0.0,
+        "mae": null,
+        "mae_by_bucket": {}
+      },
+      "metrics": {
+        "iv_bucket_mae": {
+          "error": 0.0,
+          "k": 1.0,
+          "tolerance": 0.05,
+          "weight": 1.0
+        }
+      },
+      "status": "not_available",
+      "weight": 0.3
+    },
+    "spot_iv_coupling": {
+      "meta": {
+        "corr_live": null,
+        "corr_synth": null
+      },
+      "metrics": {
+        "corr_spot_div_diff": {
+          "error": 0.0,
+          "k": 1.0,
+          "tolerance": 0.3,
+          "weight": 1.0
+        }
+      },
+      "status": "not_available",
+      "weight": 0.2
+    },
+    "strategy_pnl_parity": {
+      "meta": {
+        "n_strategies": 6
+      },
+      "metrics": {
+        "es_1pct_diff": {
+          "error": 0.0,
+          "k": 1.0,
+          "tolerance": 0.03,
+          "weight": 0.2
+        },
+        "ks": {
+          "error": 0.0,
+          "k": 1.0,
+          "tolerance": 0.2,
+          "weight": 0.2
+        },
+        "max_dd_diff": {
+          "error": 0.0,
+          "k": 1.0,
+          "tolerance": 0.1,
+          "weight": 0.1
+        },
+        "return_quantile_diff": {
+          "error": 0.0,
+          "k": 1.0,
+          "tolerance": 0.02,
+          "weight": 0.5
+        }
+      },
+      "status": "not_available",
+      "weight": 0.3
+    },
+    "underlying_returns": {
+      "meta": {
+        "quantile_diffs": {
+          "q05": 0.0,
+          "q25": 0.0,
+          "q50": 0.0,
+          "q75": 0.0,
+          "q95": 0.0,
+          "q99": 0.0
+        },
+        "rv_live": null,
+        "rv_synth": null
+      },
+      "metrics": {
+        "rv_level_diff": {
+          "error": 0.0,
+          "k": 1.0,
+          "tolerance": 0.1,
+          "weight": 0.4
+        },
+        "tail_quantile_diff": {
+          "error": 0.0,
+          "k": 1.0,
+          "tolerance": 0.02,
+          "weight": 0.6
+        }
+      },
+      "status": "not_available",
+      "weight": 0.2
+    }
+  },
+  "end_ts": 1736121600,
+  "gate": "UNTRUSTED",
+  "gate_label": "UNTRUSTED",
+  "live_data_status": "ok",
+  "market_live_meta": {
+    "ds_class": "LiveDeribitDataSource",
+    "margin_type": "linear",
+    "settlement_ccy": "USDC",
+    "type": "live_replay",
+    "underlying": "BTC"
+  },
+  "market_synth_meta": {
+    "cfg_class": "CallSimulationConfig",
+    "type": "synthetic_replay",
+    "underlying": "BTC"
+  },
+  "notes": [
+    "MVP: yardstick strategies + deterministic scoring"
+  ],
+  "overall_score": 0.0,
+  "per_strategy": {
+    "Calendar": {
+      "live_metrics": {
+        "avg_trade_return": 0.0,
+        "es_1pct": 0.0,
+        "max_drawdown": 0.0,
+        "median_trade_return": 0.0,
+        "profit_factor": 0.0,
+        "var_1pct": 0.0,
+        "win_rate": 0.0,
+        "worst_trade_return": 0.0
+      },
+      "parity_metrics": {
+        "ks": 0.0,
+        "n_live": 0,
+        "n_synth": 0,
+        "quantile_diffs": {
+          "q05": 0.0,
+          "q25": 0.0,
+          "q50": 0.0,
+          "q75": 0.0,
+          "q95": 0.0,
+          "q99": 0.0
+        }
+      },
+      "synthetic_metrics": {
+        "avg_trade_return": 0.0,
+        "es_1pct": 0.0,
+        "max_drawdown": 0.0,
+        "median_trade_return": 0.0,
+        "profit_factor": 0.0,
+        "var_1pct": 0.0,
+        "win_rate": 0.0,
+        "worst_trade_return": 0.0
+      }
+    },
+    "CallDebitSpread": {
+      "live_metrics": {
+        "avg_trade_return": 0.0,
+        "es_1pct": 0.0,
+        "max_drawdown": 0.0,
+        "median_trade_return": 0.0,
+        "profit_factor": 0.0,
+        "var_1pct": 0.0,
+        "win_rate": 0.0,
+        "worst_trade_return": 0.0
+      },
+      "parity_metrics": {
+        "ks": 0.0,
+        "n_live": 0,
+        "n_synth": 0,
+        "quantile_diffs": {
+          "q05": 0.0,
+          "q25": 0.0,
+          "q50": 0.0,
+          "q75": 0.0,
+          "q95": 0.0,
+          "q99": 0.0
+        }
+      },
+      "synthetic_metrics": {
+        "avg_trade_return": 0.0,
+        "es_1pct": 0.0,
+        "max_drawdown": 0.0,
+        "median_trade_return": 0.0,
+        "profit_factor": 0.0,
+        "var_1pct": 0.0,
+        "win_rate": 0.0,
+        "worst_trade_return": 0.0
+      }
+    },
+    "CashSecuredPut": {
+      "live_metrics": {
+        "avg_trade_return": 0.0,
+        "es_1pct": 0.0,
+        "max_drawdown": 0.0,
+        "median_trade_return": 0.0,
+        "profit_factor": 0.0,
+        "var_1pct": 0.0,
+        "win_rate": 0.0,
+        "worst_trade_return": 0.0
+      },
+      "parity_metrics": {
+        "ks": 0.0,
+        "n_live": 0,
+        "n_synth": 0,
+        "quantile_diffs": {
+          "q05": 0.0,
+          "q25": 0.0,
+          "q50": 0.0,
+          "q75": 0.0,
+          "q95": 0.0,
+          "q99": 0.0
+        }
+      },
+      "synthetic_metrics": {
+        "avg_trade_return": 0.0,
+        "es_1pct": 0.0,
+        "max_drawdown": 0.0,
+        "median_trade_return": 0.0,
+        "profit_factor": 0.0,
+        "var_1pct": 0.0,
+        "win_rate": 0.0,
+        "worst_trade_return": 0.0
+      }
+    },
+    "CoveredCall": {
+      "live_metrics": {
+        "avg_trade_return": 0.0,
+        "es_1pct": 0.0,
+        "max_drawdown": 0.0,
+        "median_trade_return": 0.0,
+        "profit_factor": 0.0,
+        "var_1pct": 0.0,
+        "win_rate": 0.0,
+        "worst_trade_return": 0.0
+      },
+      "parity_metrics": {
+        "ks": 0.0,
+        "n_live": 0,
+        "n_synth": 0,
+        "quantile_diffs": {
+          "q05": 0.0,
+          "q25": 0.0,
+          "q50": 0.0,
+          "q75": 0.0,
+          "q95": 0.0,
+          "q99": 0.0
+        }
+      },
+      "synthetic_metrics": {
+        "avg_trade_return": 0.0,
+        "es_1pct": 0.0,
+        "max_drawdown": 0.0,
+        "median_trade_return": 0.0,
+        "profit_factor": 0.0,
+        "var_1pct": 0.0,
+        "win_rate": 0.0,
+        "worst_trade_return": 0.0
+      }
+    },
+    "PutCreditSpread": {
+      "live_metrics": {
+        "avg_trade_return": 0.0,
+        "es_1pct": 0.0,
+        "max_drawdown": 0.0,
+        "median_trade_return": 0.0,
+        "profit_factor": 0.0,
+        "var_1pct": 0.0,
+        "win_rate": 0.0,
+        "worst_trade_return": 0.0
+      },
+      "parity_metrics": {
+        "ks": 0.0,
+        "n_live": 0,
+        "n_synth": 0,
+        "quantile_diffs": {
+          "q05": 0.0,
+          "q25": 0.0,
+          "q50": 0.0,
+          "q75": 0.0,
+          "q95": 0.0,
+          "q99": 0.0
+        }
+      },
+      "synthetic_metrics": {
+        "avg_trade_return": 0.0,
+        "es_1pct": 0.0,
+        "max_drawdown": 0.0,
+        "median_trade_return": 0.0,
+        "profit_factor": 0.0,
+        "var_1pct": 0.0,
+        "win_rate": 0.0,
+        "worst_trade_return": 0.0
+      }
+    },
+    "ShortStrangle": {
+      "live_metrics": {
+        "avg_trade_return": 0.0,
+        "es_1pct": 0.0,
+        "max_drawdown": 0.0,
+        "median_trade_return": 0.0,
+        "profit_factor": 0.0,
+        "var_1pct": 0.0,
+        "win_rate": 0.0,
+        "worst_trade_return": 0.0
+      },
+      "parity_metrics": {
+        "ks": 0.0,
+        "n_live": 0,
+        "n_synth": 0,
+        "quantile_diffs": {
+          "q05": 0.0,
+          "q25": 0.0,
+          "q50": 0.0,
+          "q75": 0.0,
+          "q95": 0.0,
+          "q99": 0.0
+        }
+      },
+      "synthetic_metrics": {
+        "avg_trade_return": 0.0,
+        "es_1pct": 0.0,
+        "max_drawdown": 0.0,
+        "median_trade_return": 0.0,
+        "profit_factor": 0.0,
+        "var_1pct": 0.0,
+        "win_rate": 0.0,
+        "worst_trade_return": 0.0
+      }
+    }
+  },
+  "run_id": "20251220_002927",
+  "start_ts": 1735689600,
+  "strategy_parity": {},
+  "timestamp": "2025-12-20T00:29:27.380772+00:00",
+  "underlying": "BTC"
+}
\ No newline at end of file
diff --git a/docs/FIDELITY_BTC_latest.md b/docs/FIDELITY_BTC_latest.md
new file mode 100644
index 0000000..8cb6960
--- /dev/null
+++ b/docs/FIDELITY_BTC_latest.md
@@ -0,0 +1,252 @@
+# Synthetic Fidelity Report
+
+- Run ID: 20251220_002927
+- Timestamp (UTC): 2025-12-20T00:29:27.380772+00:00
+- Gate: **UNTRUSTED**
+
+## Scores
+
+- Overall: **0.0**
+- iv_surface_level: 100.0
+- spot_iv_coupling: 100.0
+- strategy_pnl_parity: 100.0
+- underlying_returns: 100.0
+
+## Market Meta
+
+### Live
+```json
+{
+  "ds_class": "LiveDeribitDataSource",
+  "margin_type": "linear",
+  "settlement_ccy": "USDC",
+  "type": "live_replay",
+  "underlying": "BTC"
+}
+```
+
+### Synthetic
+```json
+{
+  "cfg_class": "CallSimulationConfig",
+  "type": "synthetic_replay",
+  "underlying": "BTC"
+}
+```
+
+## Strategy Parity
+
+```json
+{
+  "Calendar": {
+    "live_metrics": {
+      "avg_trade_return": 0.0,
+      "es_1pct": 0.0,
+      "max_drawdown": 0.0,
+      "median_trade_return": 0.0,
+      "profit_factor": 0.0,
+      "var_1pct": 0.0,
+      "win_rate": 0.0,
+      "worst_trade_return": 0.0
+    },
+    "parity_metrics": {
+      "ks": 0.0,
+      "n_live": 0,
+      "n_synth": 0,
+      "quantile_diffs": {
+        "q05": 0.0,
+        "q25": 0.0,
+        "q50": 0.0,
+        "q75": 0.0,
+        "q95": 0.0,
+        "q99": 0.0
+      }
+    },
+    "synthetic_metrics": {
+      "avg_trade_return": 0.0,
+      "es_1pct": 0.0,
+      "max_drawdown": 0.0,
+      "median_trade_return": 0.0,
+      "profit_factor": 0.0,
+      "var_1pct": 0.0,
+      "win_rate": 0.0,
+      "worst_trade_return": 0.0
+    }
+  },
+  "CallDebitSpread": {
+    "live_metrics": {
+      "avg_trade_return": 0.0,
+      "es_1pct": 0.0,
+      "max_drawdown": 0.0,
+      "median_trade_return": 0.0,
+      "profit_factor": 0.0,
+      "var_1pct": 0.0,
+      "win_rate": 0.0,
+      "worst_trade_return": 0.0
+    },
+    "parity_metrics": {
+      "ks": 0.0,
+      "n_live": 0,
+      "n_synth": 0,
+      "quantile_diffs": {
+        "q05": 0.0,
+        "q25": 0.0,
+        "q50": 0.0,
+        "q75": 0.0,
+        "q95": 0.0,
+        "q99": 0.0
+      }
+    },
+    "synthetic_metrics": {
+      "avg_trade_return": 0.0,
+      "es_1pct": 0.0,
+      "max_drawdown": 0.0,
+      "median_trade_return": 0.0,
+      "profit_factor": 0.0,
+      "var_1pct": 0.0,
+      "win_rate": 0.0,
+      "worst_trade_return": 0.0
+    }
+  },
+  "CashSecuredPut": {
+    "live_metrics": {
+      "avg_trade_return": 0.0,
+      "es_1pct": 0.0,
+      "max_drawdown": 0.0,
+      "median_trade_return": 0.0,
+      "profit_factor": 0.0,
+      "var_1pct": 0.0,
+      "win_rate": 0.0,
+      "worst_trade_return": 0.0
+    },
+    "parity_metrics": {
+      "ks": 0.0,
+      "n_live": 0,
+      "n_synth": 0,
+      "quantile_diffs": {
+        "q05": 0.0,
+        "q25": 0.0,
+        "q50": 0.0,
+        "q75": 0.0,
+        "q95": 0.0,
+        "q99": 0.0
+      }
+    },
+    "synthetic_metrics": {
+      "avg_trade_return": 0.0,
+      "es_1pct": 0.0,
+      "max_drawdown": 0.0,
+      "median_trade_return": 0.0,
+      "profit_factor": 0.0,
+      "var_1pct": 0.0,
+      "win_rate": 0.0,
+      "worst_trade_return": 0.0
+    }
+  },
+  "CoveredCall": {
+    "live_metrics": {
+      "avg_trade_return": 0.0,
+      "es_1pct": 0.0,
+      "max_drawdown": 0.0,
+      "median_trade_return": 0.0,
+      "profit_factor": 0.0,
+      "var_1pct": 0.0,
+      "win_rate": 0.0,
+      "worst_trade_return": 0.0
+    },
+    "parity_metrics": {
+      "ks": 0.0,
+      "n_live": 0,
+      "n_synth": 0,
+      "quantile_diffs": {
+        "q05": 0.0,
+        "q25": 0.0,
+        "q50": 0.0,
+        "q75": 0.0,
+        "q95": 0.0,
+        "q99": 0.0
+      }
+    },
+    "synthetic_metrics": {
+      "avg_trade_return": 0.0,
+      "es_1pct": 0.0,
+      "max_drawdown": 0.0,
+      "median_trade_return": 0.0,
+      "profit_factor": 0.0,
+      "var_1pct": 0.0,
+      "win_rate": 0.0,
+      "worst_trade_return": 0.0
+    }
+  },
+  "PutCreditSpread": {
+    "live_metrics": {
+      "avg_trade_return": 0.0,
+      "es_1pct": 0.0,
+      "max_drawdown": 0.0,
+      "median_trade_return": 0.0,
+      "profit_factor": 0.0,
+      "var_1pct": 0.0,
+      "win_rate": 0.0,
+      "worst_trade_return": 0.0
+    },
+    "parity_metrics": {
+      "ks": 0.0,
+      "n_live": 0,
+      "n_synth": 0,
+      "quantile_diffs": {
+        "q05": 0.0,
+        "q25": 0.0,
+        "q50": 0.0,
+        "q75": 0.0,
+        "q95": 0.0,
+        "q99": 0.0
+      }
+    },
+    "synthetic_metrics": {
+      "avg_trade_return": 0.0,
+      "es_1pct": 0.0,
+      "max_drawdown": 0.0,
+      "median_trade_return": 0.0,
+      "profit_factor": 0.0,
+      "var_1pct": 0.0,
+      "win_rate": 0.0,
+      "worst_trade_return": 0.0
+    }
+  },
+  "ShortStrangle": {
+    "live_metrics": {
+      "avg_trade_return": 0.0,
+      "es_1pct": 0.0,
+      "max_drawdown": 0.0,
+      "median_trade_return": 0.0,
+      "profit_factor": 0.0,
+      "var_1pct": 0.0,
+      "win_rate": 0.0,
+      "worst_trade_return": 0.0
+    },
+    "parity_metrics": {
+      "ks": 0.0,
+      "n_live": 0,
+      "n_synth": 0,
+      "quantile_diffs": {
+        "q05": 0.0,
+        "q25": 0.0,

TRUNCATED
