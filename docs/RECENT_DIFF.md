# Recent Diff

generated_at_utc: 2025-12-22T22:17:03Z
branch: main
head_sha: 99220a3df19cd3ca0536b87aec37b15d3600f8f7
base: origin/main

## git log --oneline -n 25
99220a3 Update context-pack artifacts
2d60e1f Auto-commit 2025-12-22T20:57:23Z
382ac49 chore: refresh generated docs
3525f55 chore: refresh latest artifacts
ccfd131 Auto-commit 2025-12-22T16:38:19Z
27408fc Ops health per-underlying gates + facts resolver
0d18785 Update roadmap to prioritize measurement integrity and a fidelity gate
0fdaf3e rep
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

## git diff --stat origin/main..HEAD
 .ci_bump                                 |    3 -
 Makefile                                 |   29 +-
 POEM.md                                  |    4 +
 ROADMAP_BACKLOG.md                       |    5 +
 docs/FIDELITY_BTC_latest.json            |  420 +---
 docs/FIDELITY_BTC_latest.md              |  305 +--
 docs/OPS_HEALTH_latest.json              |   20 +-
 docs/RECENT_DIFF.md                      | 3798 ++++++++++++++---------------
 docs/RECENT_DIFF_latest.md               |   37 +
 docs/REPO_MANIFEST.json                  |  351 +--
 docs/REPO_MANIFEST_latest.json           | 3913 ++++++++++++++++++++++++++++++
 docs/REPO_MANIFEST_latest.md             |    3 +
 docs/ROADMAP_BACKLOG_latest.md           |    5 +
 docs/TEST_SUMMARY_latest.txt             |    4 +-
 docs/supervisor-loop.md                  |   57 -
 scripts/capture_pytest_summary.sh        |   79 +-
 scripts/gen_ops_health_latest.py         |   70 +-
 scripts/gen_recent_diff.sh               |  115 +-
 scripts/gen_repo_manifest.py             |  258 +-
 scripts/gen_repo_manifest_md.py          |   70 -
 scripts/gen_roadmap_latest.sh            |   27 -
 scripts/gen_test_summary_latest.sh       |   29 -
 scripts/push_context_pack_to_drive.sh    |   97 +-
 src/healthcheck.py                       |   66 +-
 src/ops/__init__.py                      |    3 -
 src/supervisor/app.py                    |  813 ++-----
 src/supervisor/config.py                 |    7 -
 src/supervisor/loop/__init__.py          |    6 -
 src/supervisor/loop/arbiter.py           |   43 -
 src/supervisor/loop/fixers.py            |  111 -
 src/supervisor/loop/optimist.py          |   30 -
 src/supervisor/loop/policy.py            |   88 -
 src/supervisor/loop/policy_defaults.json |    9 -
 src/supervisor/loop/skeptic.py           |   20 -
 src/supervisor/loop/types.py             |   31 -
 src/supervisor/models.py                 |   78 +-
 src/supervisor/policy.py                 |    2 +-
 src/supervisor/redact.py                 |   19 +-
 src/supervisor/telegram_notify.py        |    2 +-
 src/web/dashboard.py                     |    8 +-
 tests/supervisor/test_loop_invariants.py |  243 --
 tests/supervisor/test_loop_limits.py     |  217 --
 tests/test_context_pack.py               |  104 +-
 tests/test_context_pack_ops_health.py    |   70 +
 tests/test_healthcheck_basic.py          |   19 +
 tests/test_healthcheck_config.py         |   53 +-
 tests/test_supervisor.py                 |    8 +-
 47 files changed, 7075 insertions(+), 4674 deletions(-)

## git diff origin/main..HEAD
diff --git a/.ci_bump b/.ci_bump
deleted file mode 100644
index ae63d90..0000000
--- a/.ci_bump
+++ /dev/null
@@ -1,3 +0,0 @@
-
-ci bump 2025-12-22T18:47:34Z
-ci bump 2025-12-22T18:59:04Z
diff --git a/Makefile b/Makefile
index 8f635ba..5bf1a75 100644
--- a/Makefile
+++ b/Makefile
@@ -1,15 +1,14 @@
-.PHONY: context-pack context-pack-extras context-pack-all context-pack-push
-
-context-pack:
-	./scripts/gen_repo_manifest.py
-	./scripts/gen_repo_manifest_md.py
-	./scripts/gen_recent_diff.sh
-
-context-pack-extras:
-	./scripts/gen_roadmap_latest.sh
-	./scripts/gen_test_summary_latest.sh
-
-context-pack-all: context-pack context-pack-extras
-
-context-pack-push: context-pack-all
-	CONTEXT_PACK_PUSH_DIRECT=1 ./scripts/push_context_pack_to_drive.sh
+extras:
+	python3 scripts/gen_ops_health_latest.py
+	@if [ ! -f docs/TEST_SUMMARY_latest.txt ]; then \
+		printf "%s\n%s\n" "$$(date -u +%Y-%m-%dT%H:%MZ)" "pytest summary unavailable" > docs/TEST_SUMMARY_latest.txt; \
+	fi
+	cp ROADMAP_BACKLOG.md docs/ROADMAP_BACKLOG_latest.md
+	@echo "Updated docs/ROADMAP_BACKLOG_latest.md (upload handled externally)"
+
+context-pack: extras
+	python3 scripts/gen_repo_manifest.py
+	bash scripts/gen_recent_diff.sh
+	python3 scripts/gen_fidelity_latest_docs.py
+
+context-pack-push: context-pack
diff --git a/POEM.md b/POEM.md
index 2744d6e..ac8a4d2 100644
--- a/POEM.md
+++ b/POEM.md
@@ -20,3 +20,7 @@ The agent learns, the agent grows.
 So here's to code that trades with care,
 A faithful servant, always there.
 
+
+
+
+
diff --git a/ROADMAP_BACKLOG.md b/ROADMAP_BACKLOG.md
index 131a69a..74b0dcd 100644
--- a/ROADMAP_BACKLOG.md
+++ b/ROADMAP_BACKLOG.md
@@ -815,6 +815,11 @@ The system supports three levels of automation, rolled out incrementally:
 ## Changelog (auto)
 - (entries appended newest-first)
 
+- 2025-12-22T19:11Z [COPILOT] sha=382ac49
+  - Summary: Ops health single-truth: gate_overall drives overall_status/summary when present; checks only block if can_trade=false; fail-closed on gate eval error when gate modes enabled
+  - Tests: 825 passed, 5 skipped (full suite)
+  - Endpoints: none
+  - Context-pack: uploaded (no)
 - 2025-12-22T16:55Z [COPILOT] sha=fa5a78a
   - Summary: Refresh TEST_SUMMARY_latest.txt after full-suite run; Refresh context-pack latest artifacts prior to push
   - Tests: 822 passed, 5 skipped, 53 warnings in 335.18s (0:05:35)
diff --git a/docs/FIDELITY_BTC_latest.json b/docs/FIDELITY_BTC_latest.json
index ff44c7f..135e9e4 100644
--- a/docs/FIDELITY_BTC_latest.json
+++ b/docs/FIDELITY_BTC_latest.json
@@ -1,118 +1,9 @@
 {
   "component_scores": {
-    "iv_surface_level": 100.0,
-    "spot_iv_coupling": 100.0,
-    "strategy_pnl_parity": 100.00000000000001,
+    "strategy_pnl_parity": 100.0,
     "underlying_returns": 100.0
   },
-  "component_status": {
-    "iv_surface_level": "not_available",
-    "spot_iv_coupling": "not_available",
-    "strategy_pnl_parity": "not_available",
-    "underlying_returns": "not_available"
-  },
-  "components": {
-    "iv_surface_level": {
-      "meta": {
-        "coverage": 0.0,
-        "mae": null,
-        "mae_by_bucket": {}
-      },
-      "metrics": {
-        "iv_bucket_mae": {
-          "error": 0.0,
-          "k": 1.0,
-          "tolerance": 0.05,
-          "weight": 1.0
-        }
-      },
-      "status": "not_available",
-      "weight": 0.3
-    },
-    "spot_iv_coupling": {
-      "meta": {
-        "corr_live": null,
-        "corr_synth": null
-      },
-      "metrics": {
-        "corr_spot_div_diff": {
-          "error": 0.0,
-          "k": 1.0,
-          "tolerance": 0.3,
-          "weight": 1.0
-        }
-      },
-      "status": "not_available",
-      "weight": 0.2
-    },
-    "strategy_pnl_parity": {
-      "meta": {
-        "n_strategies": 6
-      },
-      "metrics": {
-        "es_1pct_diff": {
-          "error": 0.0,
-          "k": 1.0,
-          "tolerance": 0.03,
-          "weight": 0.2
-        },
-        "ks": {
-          "error": 0.0,
-          "k": 1.0,
-          "tolerance": 0.2,
-          "weight": 0.2
-        },
-        "max_dd_diff": {
-          "error": 0.0,
-          "k": 1.0,
-          "tolerance": 0.1,
-          "weight": 0.1
-        },
-        "return_quantile_diff": {
-          "error": 0.0,
-          "k": 1.0,
-          "tolerance": 0.02,
-          "weight": 0.5
-        }
-      },
-      "status": "not_available",
-      "weight": 0.3
-    },
-    "underlying_returns": {
-      "meta": {
-        "quantile_diffs": {
-          "q05": 0.0,
-          "q25": 0.0,
-          "q50": 0.0,
-          "q75": 0.0,
-          "q95": 0.0,
-          "q99": 0.0
-        },
-        "rv_live": null,
-        "rv_synth": null
-      },
-      "metrics": {
-        "rv_level_diff": {
-          "error": 0.0,
-          "k": 1.0,
-          "tolerance": 0.1,
-          "weight": 0.4
-        },
-        "tail_quantile_diff": {
-          "error": 0.0,
-          "k": 1.0,
-          "tolerance": 0.02,
-          "weight": 0.6
-        }
-      },
-      "status": "not_available",
-      "weight": 0.2
-    }
-  },
-  "end_ts": 1736121600,
-  "gate": "UNTRUSTED",
-  "gate_label": "UNTRUSTED",
-  "live_data_status": "ok",
+  "gate": "TRUSTED",
   "market_live_meta": {
     "ds_class": "LiveDeribitDataSource",
     "margin_type": "linear",
@@ -125,225 +16,116 @@
     "type": "synthetic_replay",
     "underlying": "BTC"
   },
-  "notes": [
-    "MVP: yardstick strategies + deterministic scoring"
-  ],
-  "overall_score": 0.0,
-  "per_strategy": {
-    "Calendar": {
-      "live_metrics": {
-        "avg_trade_return": 0.0,
-        "es_1pct": 0.0,
-        "max_drawdown": 0.0,
-        "median_trade_return": 0.0,
-        "profit_factor": 0.0,
-        "var_1pct": 0.0,
-        "win_rate": 0.0,
-        "worst_trade_return": 0.0
-      },
-      "parity_metrics": {
-        "ks": 0.0,
-        "n_live": 0,
-        "n_synth": 0,
-        "quantile_diffs": {
-          "q05": 0.0,
-          "q25": 0.0,
-          "q50": 0.0,
-          "q75": 0.0,
-          "q95": 0.0,
-          "q99": 0.0
+  "overall_score": 100.0,
+  "run_id": "20251219_234717",
+  "strategy_parity": {
+    "decision_times": [
+      "2025-12-07T00:00:00+00:00",
+      "2025-12-08T00:00:00+00:00",
+      "2025-12-09T00:00:00+00:00",
+      "2025-12-10T00:00:00+00:00",
+      "2025-12-11T00:00:00+00:00",
+      "2025-12-12T00:00:00+00:00",
+      "2025-12-13T00:00:00+00:00",
+      "2025-12-14T00:00:00+00:00",
+      "2025-12-15T00:00:00+00:00",
+      "2025-12-16T00:00:00+00:00",
+      "2025-12-17T00:00:00+00:00",
+      "2025-12-18T00:00:00+00:00",
+      "2025-12-19T00:00:00+00:00"
+    ],
+    "strategies": [
+      {
+        "live": {
+          "notes": "P0 placeholder (execution not implemented)",
+          "num_trades": 0,
+          "spot_first": 0.0,
+          "spot_last": 0.0
+        },
+        "name": "covered_call",
+        "synthetic": {
+          "notes": "P0 placeholder (execution not implemented)",
+          "num_trades": 0,
+          "spot_first": 0.0,
+          "spot_last": 85758.41
         }
       },
-      "synthetic_metrics": {
-        "avg_trade_return": 0.0,
-        "es_1pct": 0.0,
-        "max_drawdown": 0.0,
-        "median_trade_return": 0.0,
-        "profit_factor": 0.0,
-        "var_1pct": 0.0,
-        "win_rate": 0.0,
-        "worst_trade_return": 0.0
-      }
-    },
-    "CallDebitSpread": {
-      "live_metrics": {
-        "avg_trade_return": 0.0,
-        "es_1pct": 0.0,
-        "max_drawdown": 0.0,
-        "median_trade_return": 0.0,
-        "profit_factor": 0.0,
-        "var_1pct": 0.0,
-        "win_rate": 0.0,
-        "worst_trade_return": 0.0
-      },
-      "parity_metrics": {
-        "ks": 0.0,
-        "n_live": 0,
-        "n_synth": 0,
-        "quantile_diffs": {
-          "q05": 0.0,
-          "q25": 0.0,
-          "q50": 0.0,
-          "q75": 0.0,
-          "q95": 0.0,
-          "q99": 0.0
+      {
+        "live": {
+          "notes": "P0 placeholder (execution not implemented)",
+          "num_trades": 0,
+          "spot_first": 0.0,
+          "spot_last": 0.0
+        },
+        "name": "cash_secured_put",
+        "synthetic": {
+          "notes": "P0 placeholder (execution not implemented)",
+          "num_trades": 0,
+          "spot_first": 0.0,
+          "spot_last": 85758.41
         }
       },
-      "synthetic_metrics": {
-        "avg_trade_return": 0.0,
-        "es_1pct": 0.0,
-        "max_drawdown": 0.0,
-        "median_trade_return": 0.0,
-        "profit_factor": 0.0,
-        "var_1pct": 0.0,
-        "win_rate": 0.0,
-        "worst_trade_return": 0.0
-      }
-    },
-    "CashSecuredPut": {
-      "live_metrics": {
-        "avg_trade_return": 0.0,
-        "es_1pct": 0.0,
-        "max_drawdown": 0.0,
-        "median_trade_return": 0.0,
-        "profit_factor": 0.0,
-        "var_1pct": 0.0,
-        "win_rate": 0.0,
-        "worst_trade_return": 0.0
-      },
-      "parity_metrics": {
-        "ks": 0.0,
-        "n_live": 0,
-        "n_synth": 0,
-        "quantile_diffs": {
-          "q05": 0.0,
-          "q25": 0.0,
-          "q50": 0.0,
-          "q75": 0.0,
-          "q95": 0.0,
-          "q99": 0.0
+      {
+        "live": {
+          "notes": "P0 placeholder (execution not implemented)",
+          "num_trades": 0,
+          "spot_first": 0.0,
+          "spot_last": 0.0
+        },
+        "name": "short_strangle",
+        "synthetic": {
+          "notes": "P0 placeholder (execution not implemented)",
+          "num_trades": 0,
+          "spot_first": 0.0,
+          "spot_last": 85758.41
         }
       },
-      "synthetic_metrics": {
-        "avg_trade_return": 0.0,
-        "es_1pct": 0.0,
-        "max_drawdown": 0.0,
-        "median_trade_return": 0.0,
-        "profit_factor": 0.0,
-        "var_1pct": 0.0,
-        "win_rate": 0.0,
-        "worst_trade_return": 0.0
-      }
-    },
-    "CoveredCall": {
-      "live_metrics": {
-        "avg_trade_return": 0.0,
-        "es_1pct": 0.0,
-        "max_drawdown": 0.0,
-        "median_trade_return": 0.0,
-        "profit_factor": 0.0,
-        "var_1pct": 0.0,
-        "win_rate": 0.0,
-        "worst_trade_return": 0.0
-      },
-      "parity_metrics": {
-        "ks": 0.0,
-        "n_live": 0,
-        "n_synth": 0,
-        "quantile_diffs": {
-          "q05": 0.0,
-          "q25": 0.0,
-          "q50": 0.0,
-          "q75": 0.0,
-          "q95": 0.0,
-          "q99": 0.0
+      {
+        "live": {
+          "notes": "P0 placeholder (execution not implemented)",
+          "num_trades": 0,
+          "spot_first": 0.0,
+          "spot_last": 0.0
+        },
+        "name": "put_spread_credit",
+        "synthetic": {
+          "notes": "P0 placeholder (execution not implemented)",
+          "num_trades": 0,
+          "spot_first": 0.0,
+          "spot_last": 85758.41
         }
       },
-      "synthetic_metrics": {
-        "avg_trade_return": 0.0,
-        "es_1pct": 0.0,
-        "max_drawdown": 0.0,
-        "median_trade_return": 0.0,
-        "profit_factor": 0.0,
-        "var_1pct": 0.0,
-        "win_rate": 0.0,
-        "worst_trade_return": 0.0
-      }
-    },
-    "PutCreditSpread": {
-      "live_metrics": {
-        "avg_trade_return": 0.0,
-        "es_1pct": 0.0,
-        "max_drawdown": 0.0,
-        "median_trade_return": 0.0,
-        "profit_factor": 0.0,
-        "var_1pct": 0.0,
-        "win_rate": 0.0,
-        "worst_trade_return": 0.0
-      },
-      "parity_metrics": {
-        "ks": 0.0,
-        "n_live": 0,
-        "n_synth": 0,
-        "quantile_diffs": {
-          "q05": 0.0,
-          "q25": 0.0,
-          "q50": 0.0,
-          "q75": 0.0,
-          "q95": 0.0,
-          "q99": 0.0
+      {
+        "live": {
+          "notes": "P0 placeholder (execution not implemented)",
+          "num_trades": 0,
+          "spot_first": 0.0,
+          "spot_last": 0.0
+        },
+        "name": "call_spread_debit",
+        "synthetic": {
+          "notes": "P0 placeholder (execution not implemented)",
+          "num_trades": 0,
+          "spot_first": 0.0,
+          "spot_last": 85758.41
         }
       },
-      "synthetic_metrics": {
-        "avg_trade_return": 0.0,
-        "es_1pct": 0.0,
-        "max_drawdown": 0.0,
-        "median_trade_return": 0.0,
-        "profit_factor": 0.0,
-        "var_1pct": 0.0,
-        "win_rate": 0.0,
-        "worst_trade_return": 0.0
-      }
-    },
-    "ShortStrangle": {
-      "live_metrics": {
-        "avg_trade_return": 0.0,
-        "es_1pct": 0.0,
-        "max_drawdown": 0.0,
-        "median_trade_return": 0.0,
-        "profit_factor": 0.0,
-        "var_1pct": 0.0,
-        "win_rate": 0.0,
-        "worst_trade_return": 0.0
-      },
-      "parity_metrics": {
-        "ks": 0.0,
-        "n_live": 0,
-        "n_synth": 0,
-        "quantile_diffs": {
-          "q05": 0.0,
-          "q25": 0.0,
-          "q50": 0.0,
-          "q75": 0.0,
-          "q95": 0.0,
-          "q99": 0.0
+      {
+        "live": {
+          "notes": "P0 placeholder (execution not implemented)",
+          "num_trades": 0,
+          "spot_first": 0.0,
+          "spot_last": 0.0
+        },
+        "name": "calendar",
+        "synthetic": {
+          "notes": "P0 placeholder (execution not implemented)",
+          "num_trades": 0,
+          "spot_first": 0.0,
+          "spot_last": 85758.41
         }
-      },
-      "synthetic_metrics": {
-        "avg_trade_return": 0.0,
-        "es_1pct": 0.0,
-        "max_drawdown": 0.0,
-        "median_trade_return": 0.0,
-        "profit_factor": 0.0,
-        "var_1pct": 0.0,
-        "win_rate": 0.0,
-        "worst_trade_return": 0.0
       }
-    }
+    ]
   },
-  "run_id": "20251220_002927",
-  "start_ts": 1735689600,
-  "strategy_parity": {},
-  "timestamp": "2025-12-20T00:29:27.380772+00:00",
-  "underlying": "BTC"
+  "timestamp": "2025-12-19T23:47:17.593360+00:00"
 }
\ No newline at end of file
diff --git a/docs/FIDELITY_BTC_latest.md b/docs/FIDELITY_BTC_latest.md
index 8cb6960..68f3de9 100644
--- a/docs/FIDELITY_BTC_latest.md
+++ b/docs/FIDELITY_BTC_latest.md
@@ -1,14 +1,12 @@
 # Synthetic Fidelity Report
 
-- Run ID: 20251220_002927
-- Timestamp (UTC): 2025-12-20T00:29:27.380772+00:00
-- Gate: **UNTRUSTED**
+- Run ID: 20251219_234717
+- Timestamp (UTC): 2025-12-19T23:47:17.593360+00:00
+- Gate: **TRUSTED**
 
 ## Scores
 
-- Overall: **0.0**
-- iv_surface_level: 100.0
-- spot_iv_coupling: 100.0
+- Overall: **100.0**
 - strategy_pnl_parity: 100.0
 - underlying_returns: 100.0
 
@@ -34,219 +32,116 @@
 }
 ```
 
-## Strategy Parity
+## Strategy Parity (P0 placeholder)
 
 ```json
 {
-  "Calendar": {
-    "live_metrics": {
-      "avg_trade_return": 0.0,
-      "es_1pct": 0.0,
-      "max_drawdown": 0.0,
-      "median_trade_return": 0.0,
-      "profit_factor": 0.0,
-      "var_1pct": 0.0,
-      "win_rate": 0.0,
-      "worst_trade_return": 0.0
-    },
-    "parity_metrics": {
-      "ks": 0.0,
-      "n_live": 0,
-      "n_synth": 0,
-      "quantile_diffs": {
-        "q05": 0.0,
-        "q25": 0.0,
-        "q50": 0.0,
-        "q75": 0.0,
-        "q95": 0.0,
-        "q99": 0.0
+  "decision_times": [
+    "2025-12-07T00:00:00+00:00",
+    "2025-12-08T00:00:00+00:00",
+    "2025-12-09T00:00:00+00:00",
+    "2025-12-10T00:00:00+00:00",
+    "2025-12-11T00:00:00+00:00",
+    "2025-12-12T00:00:00+00:00",
+    "2025-12-13T00:00:00+00:00",
+    "2025-12-14T00:00:00+00:00",
+    "2025-12-15T00:00:00+00:00",
+    "2025-12-16T00:00:00+00:00",
+    "2025-12-17T00:00:00+00:00",
+    "2025-12-18T00:00:00+00:00",
+    "2025-12-19T00:00:00+00:00"
+  ],
+  "strategies": [
+    {
+      "live": {
+        "notes": "P0 placeholder (execution not implemented)",
+        "num_trades": 0,
+        "spot_first": 0.0,
+        "spot_last": 0.0
+      },
+      "name": "covered_call",
+      "synthetic": {
+        "notes": "P0 placeholder (execution not implemented)",
+        "num_trades": 0,
+        "spot_first": 0.0,
+        "spot_last": 85758.41
       }
     },
-    "synthetic_metrics": {
-      "avg_trade_return": 0.0,
-      "es_1pct": 0.0,
-      "max_drawdown": 0.0,
-      "median_trade_return": 0.0,
-      "profit_factor": 0.0,
-      "var_1pct": 0.0,
-      "win_rate": 0.0,
-      "worst_trade_return": 0.0
-    }
-  },
-  "CallDebitSpread": {
-    "live_metrics": {
-      "avg_trade_return": 0.0,
-      "es_1pct": 0.0,
-      "max_drawdown": 0.0,
-      "median_trade_return": 0.0,
-      "profit_factor": 0.0,
-      "var_1pct": 0.0,
-      "win_rate": 0.0,
-      "worst_trade_return": 0.0
-    },
-    "parity_metrics": {
-      "ks": 0.0,
-      "n_live": 0,
-      "n_synth": 0,
-      "quantile_diffs": {
-        "q05": 0.0,
-        "q25": 0.0,
-        "q50": 0.0,
-        "q75": 0.0,
-        "q95": 0.0,
-        "q99": 0.0
+    {
+      "live": {
+        "notes": "P0 placeholder (execution not implemented)",
+        "num_trades": 0,
+        "spot_first": 0.0,
+        "spot_last": 0.0
+      },
+      "name": "cash_secured_put",
+      "synthetic": {
+        "notes": "P0 placeholder (execution not implemented)",
+        "num_trades": 0,
+        "spot_first": 0.0,
+        "spot_last": 85758.41
       }
     },
-    "synthetic_metrics": {
-      "avg_trade_return": 0.0,
-      "es_1pct": 0.0,
-      "max_drawdown": 0.0,
-      "median_trade_return": 0.0,
-      "profit_factor": 0.0,
-      "var_1pct": 0.0,
-      "win_rate": 0.0,
-      "worst_trade_return": 0.0
-    }
-  },
-  "CashSecuredPut": {
-    "live_metrics": {
-      "avg_trade_return": 0.0,
-      "es_1pct": 0.0,
-      "max_drawdown": 0.0,
-      "median_trade_return": 0.0,
-      "profit_factor": 0.0,
-      "var_1pct": 0.0,
-      "win_rate": 0.0,
-      "worst_trade_return": 0.0
-    },
-    "parity_metrics": {
-      "ks": 0.0,
-      "n_live": 0,
-      "n_synth": 0,
-      "quantile_diffs": {
-        "q05": 0.0,
-        "q25": 0.0,
-        "q50": 0.0,
-        "q75": 0.0,
-        "q95": 0.0,
-        "q99": 0.0
+    {
+      "live": {
+        "notes": "P0 placeholder (execution not implemented)",
+        "num_trades": 0,
+        "spot_first": 0.0,
+        "spot_last": 0.0
+      },
+      "name": "short_strangle",
+      "synthetic": {
+        "notes": "P0 placeholder (execution not implemented)",
+        "num_trades": 0,
+        "spot_first": 0.0,
+        "spot_last": 85758.41
       }
     },
-    "synthetic_metrics": {
-      "avg_trade_return": 0.0,
-      "es_1pct": 0.0,
-      "max_drawdown": 0.0,
-      "median_trade_return": 0.0,
-      "profit_factor": 0.0,
-      "var_1pct": 0.0,
-      "win_rate": 0.0,
-      "worst_trade_return": 0.0
-    }
-  },
-  "CoveredCall": {
-    "live_metrics": {
-      "avg_trade_return": 0.0,
-      "es_1pct": 0.0,
-      "max_drawdown": 0.0,
-      "median_trade_return": 0.0,
-      "profit_factor": 0.0,
-      "var_1pct": 0.0,
-      "win_rate": 0.0,
-      "worst_trade_return": 0.0
-    },
-    "parity_metrics": {
-      "ks": 0.0,
-      "n_live": 0,
-      "n_synth": 0,
-      "quantile_diffs": {
-        "q05": 0.0,
-        "q25": 0.0,
-        "q50": 0.0,
-        "q75": 0.0,
-        "q95": 0.0,
-        "q99": 0.0
+    {
+      "live": {
+        "notes": "P0 placeholder (execution not implemented)",
+        "num_trades": 0,
+        "spot_first": 0.0,
+        "spot_last": 0.0
+      },
+      "name": "put_spread_credit",
+      "synthetic": {
+        "notes": "P0 placeholder (execution not implemented)",
+        "num_trades": 0,
+        "spot_first": 0.0,
+        "spot_last": 85758.41
       }
     },
-    "synthetic_metrics": {
-      "avg_trade_return": 0.0,
-      "es_1pct": 0.0,
-      "max_drawdown": 0.0,
-      "median_trade_return": 0.0,
-      "profit_factor": 0.0,
-      "var_1pct": 0.0,
-      "win_rate": 0.0,
-      "worst_trade_return": 0.0
-    }
-  },
-  "PutCreditSpread": {
-    "live_metrics": {
-      "avg_trade_return": 0.0,
-      "es_1pct": 0.0,
-      "max_drawdown": 0.0,
-      "median_trade_return": 0.0,
-      "profit_factor": 0.0,
-      "var_1pct": 0.0,
-      "win_rate": 0.0,
-      "worst_trade_return": 0.0
-    },
-    "parity_metrics": {
-      "ks": 0.0,
-      "n_live": 0,
-      "n_synth": 0,
-      "quantile_diffs": {
-        "q05": 0.0,
-        "q25": 0.0,
-        "q50": 0.0,
-        "q75": 0.0,
-        "q95": 0.0,
-        "q99": 0.0
+    {
+      "live": {
+        "notes": "P0 placeholder (execution not implemented)",
+        "num_trades": 0,
+        "spot_first": 0.0,
+        "spot_last": 0.0
+      },
+      "name": "call_spread_debit",
+      "synthetic": {
+        "notes": "P0 placeholder (execution not implemented)",
+        "num_trades": 0,
+        "spot_first": 0.0,
+        "spot_last": 85758.41
       }
     },
-    "synthetic_metrics": {
-      "avg_trade_return": 0.0,
-      "es_1pct": 0.0,
-      "max_drawdown": 0.0,
-      "median_trade_return": 0.0,
-      "profit_factor": 0.0,
-      "var_1pct": 0.0,
-      "win_rate": 0.0,
-      "worst_trade_return": 0.0
-    }
-  },
-  "ShortStrangle": {
-    "live_metrics": {
-      "avg_trade_return": 0.0,
-      "es_1pct": 0.0,
-      "max_drawdown": 0.0,
-      "median_trade_return": 0.0,
-      "profit_factor": 0.0,
-      "var_1pct": 0.0,
-      "win_rate": 0.0,
-      "worst_trade_return": 0.0
-    },
-    "parity_metrics": {
-      "ks": 0.0,
-      "n_live": 0,
-      "n_synth": 0,
-      "quantile_diffs": {
-        "q05": 0.0,
-        "q25": 0.0,
-        "q50": 0.0,
-        "q75": 0.0,
-        "q95": 0.0,
-        "q99": 0.0
+    {
+      "live": {
+        "notes": "P0 placeholder (execution not implemented)",
+        "num_trades": 0,
+        "spot_first": 0.0,
+        "spot_last": 0.0
+      },
+      "name": "calendar",
+      "synthetic": {
+        "notes": "P0 placeholder (execution not implemented)",
+        "num_trades": 0,
+        "spot_first": 0.0,
+        "spot_last": 85758.41
       }
-    },
-    "synthetic_metrics": {
-      "avg_trade_return": 0.0,
-      "es_1pct": 0.0,
-      "max_drawdown": 0.0,
-      "median_trade_return": 0.0,
-      "profit_factor": 0.0,
-      "var_1pct": 0.0,
-      "win_rate": 0.0,
-      "worst_trade_return": 0.0
     }
-  }
+  ]
 }
 ```
diff --git a/docs/OPS_HEALTH_latest.json b/docs/OPS_HEALTH_latest.json
index d6e6250..918be3c 100644
--- a/docs/OPS_HEALTH_latest.json
+++ b/docs/OPS_HEALTH_latest.json
@@ -1,10 +1,24 @@
 {
+  "agent_paused_due_to_health": null,
+  "cache_age_seconds": null,
+  "can_trade": false,
+  "can_trade_by_underlying": null,
+  "checked_at": null,
+  "checks": [],
+  "checks_overall": "FAIL",
+  "checks_summary": null,
   "error": {
     "message": "No module named 'src'",
     "type": "ModuleNotFoundError"
   },
-  "generated_at_utc": "2025-12-22T17:02Z",
-  "head_sha": "f6e339582a45a38469c24cc6b14a910fdfc49bdd",
+  "error_code": "OPS_HEALTH_GENERATION_ERROR",
+  "error_message": "No module named 'src'",
+  "gate_overall": null,
+  "gates": [],
+  "generated_at_utc": "2025-12-22T22:06Z",
+  "head_sha": "2d60e1f135590e2728398fadcc3049e277bebbb2",
+  "last_run_at": null,
   "overall_status": "FAIL",
-  "summary": "OPS_HEALTH_GENERATION_ERROR"
+  "summary": "OPS_HEALTH_GENERATION_ERROR",
+  "worst_severity": "FATAL"
 }
diff --git a/docs/RECENT_DIFF.md b/docs/RECENT_DIFF.md
index 8b52c51..de4abfe 100644
--- a/docs/RECENT_DIFF.md
+++ b/docs/RECENT_DIFF.md
@@ -1,14 +1,18 @@
 # Recent Diff
 
-generated_at_utc: 2025-12-22T17:02:24Z
+generated_at_utc: 2025-12-22T22:06:06Z
 branch: main
-head_sha: f6e339582a45a38469c24cc6b14a910fdfc49bdd
+head_sha: 2d60e1f135590e2728398fadcc3049e277bebbb2
 base: origin/main
 
 ## git log --oneline -n 25
-f6e3395 chore: refresh latest artifacts
-fa5a78a Auto-commit 2025-12-22T16:38:19Z
-3bc7e86 Ops health per-underlying gates + facts resolver
+2d60e1f Auto-commit 2025-12-22T20:57:23Z
+382ac49 chore: refresh generated docs
+3525f55 chore: refresh latest artifacts
+ccfd131 Auto-commit 2025-12-22T16:38:19Z
+27408fc Ops health per-underlying gates + facts resolver
+0d18785 Update roadmap to prioritize measurement integrity and a fidelity gate
+0fdaf3e rep
 522ef78 feat(fidelity): deterministic per-underlying latest + source/path metadata + tests
 e5046b2 Merge pull request #5 from speelbreaker/security-secret-hygiene-clean
 95d9b23 test: ensure bearer redaction hits regex threshold
@@ -27,1976 +31,1972 @@ ac51b37 Roadmap: define calibration acceptance criteria and fidelity gate
 ae29c66 Add build SHA indicator endpoint and UI
 bba4c05 Add fidelity suite UI + API endpoints
 ef5aa6f test
-bd1d2e7 Fix harvested calibration RV scaling
-84d4601 Block Deribit live_chain historically
-c403e9c Backtest correctness: no overlap + linear USDC + cap expiry
-e39b4a7 Finish regime IV plumbing + add provenance summary
 
 ## git diff --stat origin/main..HEAD
- .vscode/tasks.json                                 |    2 +-
- HEALTHCHECK.md                                     |  126 +-
- Makefile                                           |   14 +
- POEM.md                                            |   22 +
- README.md                                          |   41 +
- ROADMAP_BACKLOG.md                                 |   45 +
- agent_loop.py                                      |   21 +
- ...uct-minded-engineer-Update-th_1766256235012.txt |   73 -
- data/backtests/index.jsonl                         |   18 +
- .../history/20251219_234717/fidelity_report.json   |  131 -
- .../BTC/history/20251219_234717/fidelity_report.md |  147 -
- data/fidelity_runs/BTC/latest/fidelity_report.json |  131 -
- data/fidelity_runs/BTC/latest/fidelity_report.md   |  147 -
- docs/CONTEXT_PACK.md                               |  219 +-
- docs/FIDELITY_BTC_latest.json                      |  349 ++
- docs/FIDELITY_BTC_latest.md                        |  252 ++
- docs/OPS_HEALTH_latest.json                        |   10 +
- docs/RECENT_DIFF.md                                | 2002 ++++++++++
- docs/REPO_MANIFEST.json                            | 3903 ++++++++++++++++++++
- docs/ROADMAP_BACKLOG_latest.md                     |  858 +++++
- docs/TEST_SUMMARY_latest.txt                       |    2 +
- scripts/capture_pytest_summary.sh                  |   49 +
- scripts/gen_fidelity_latest_docs.py                |   94 +
- scripts/gen_ops_health_latest.py                   |  114 +
- scripts/gen_recent_diff.sh                         |   77 +
- scripts/gen_repo_manifest.py                       |  188 +
- scripts/print_fidelity_summary.py                  |   94 +
- scripts/push_context_pack_to_drive.sh              |   83 +
- scripts/roadmap_append_changelog.py                |  151 +
- scripts/run_fidelity_from_lab.py                   |   56 +
- scripts/run_fidelity_from_lab_daily.py             |   56 +
- scripts/sabotage_fidelity_drill.py                 |   42 +
- src/backtest/compare.py                            |  337 +-
- src/backtest/covered_call_simulator.py             |    7 +
- src/backtest/diff.py                               |  169 +-
- src/backtest/fidelity_store.py                     |  208 +-
- src/backtest/fidelity_suite.py                     |  404 ++
- src/backtest/live_deribit_data_source.py           |   29 +-
- src/backtest/manager.py                            |  147 +-
- src/backtest/pricing.py                            |   12 +-
- src/backtest/state_builder.py                      |   32 +-
- src/backtest/units.py                              |   58 +
- src/calibration_config.py                          |    1 +
- src/calibration_extended.py                        |    3 +
- src/calibration_update_policy.py                   |   82 +-
- src/config.py                                      |   30 +
- src/data/live_deribit_exam.py                      |   37 +
- src/db/__init__.py                                 |    1 +
- src/db/models_telegram.py                          |   21 +
- src/fidelity/canonical_strategies.py               |  194 +-
- src/fidelity/gating.py                             |   17 +
- src/fidelity/market_replay.py                      |  209 +-
- src/fidelity/ops_runner.py                         |  284 ++
- src/fidelity/reporting.py                          |    7 +
- src/fidelity/run_suite.py                          |  114 +-
- src/fidelity/scoring.py                            |   44 +-
- src/harvest_status.py                              |  271 ++
- src/healthcheck.py                                 |  364 +-
- src/ops/calibration_status.py                      |   98 +
- src/ops/facts_resolver.py                          |   89 +
- src/ops/fidelity_status.py                         |   91 +-
- src/ops/gate_factories.py                          |  442 +++
- src/ops/gates.py                                   |  112 +
- src/telegram/__init__.py                           |    1 +
- src/telegram/store.py                              |  108 +
- src/web/api_errors.py                              |   41 +
- src/web/dashboard.py                               |  313 +-
- src/web/routes_backtest.py                         |  131 +-
- src/web/routes_fidelity.py                         |   46 +
- src/web/routes_health.py                           |   37 +-
- src/web/routes_positions.py                        |   12 +
- src/web/routes_telegram.py                         |  214 ++
- src/web_app.py                                     |   18 +
- tests/test_api_calibration_run_with_policy.py      |   54 +
- tests/test_api_fidelity_endpoints.py               |   64 +
- tests/test_backtest_greg_modes.py                  |   48 +-
- tests/test_backtest_preflight.py                   |  214 ++
- tests/test_calibration_update_policy.py            |    2 +
- tests/test_context_pack.py                         |   66 +
- tests/test_fidelity_canonical_store.py             |  125 +
- tests/test_fidelity_gate_integration.py            |   84 +
- tests/test_fidelity_lab_scoring.py                 |   25 +
- tests/test_fidelity_latest_docs_generator.py       |   52 +
- tests/test_fidelity_latest_resolution.py           |   44 +-
- tests/test_fidelity_missing_close.py               |  126 +
- ..._fidelity_moneyness_fallback_and_diagnostics.py |   68 +
- tests/test_gen_ops_health_latest.py                |   61 +
- tests/test_health_and_calibration_automation.py    |   93 +-
- tests/test_healthcheck_basic.py                    |  259 +-
- tests/test_healthcheck_config.py                   |   15 +-
- tests/test_live_deribit_units.py                   |   53 +
- tests/test_ops_fidelity_clamping_and_schema.py     |  102 +
- tests/test_ops_fidelity_coverage_penalty.py        |   36 +
- tests/test_ops_health_artifact.py                  |   40 +
- tests/test_ops_health_endpoints.py                 |  809 ++++
- tests/web/expected_routes.json                     |  443 ++-
- tests/web/test_telegram_webhook.py                 |  126 +
- 97 files changed, 15627 insertions(+), 1734 deletions(-)
+ .ci_bump                                 |    3 -
+ Makefile                                 |   29 +-
+ POEM.md                                  |    4 +
+ ROADMAP_BACKLOG.md                       |    5 +
+ docs/FIDELITY_BTC_latest.json            |  420 +---
+ docs/FIDELITY_BTC_latest.md              |  305 +--
+ docs/OPS_HEALTH_latest.json              |   25 +-
+ docs/RECENT_DIFF.md                      | 1981 +--------------
+ docs/RECENT_DIFF_latest.md               |   37 +
+ docs/REPO_MANIFEST.json                  |  349 +--
+ docs/REPO_MANIFEST_latest.json           | 3913 ++++++++++++++++++++++++++++++
+ docs/REPO_MANIFEST_latest.md             |    3 +
+ docs/ROADMAP_BACKLOG_latest.md           |    5 +
+ docs/TEST_SUMMARY_latest.txt             |    4 +-
+ docs/supervisor-loop.md                  |   57 -
+ scripts/capture_pytest_summary.sh        |   79 +-
+ scripts/gen_ops_health_latest.py         |   70 +-
+ scripts/gen_recent_diff.sh               |  115 +-
+ scripts/gen_repo_manifest.py             |  258 +-
+ scripts/gen_repo_manifest_md.py          |   70 -
+ scripts/gen_roadmap_latest.sh            |   27 -
+ scripts/gen_test_summary_latest.sh       |   29 -
+ scripts/push_context_pack_to_drive.sh    |   97 +-
+ src/healthcheck.py                       |   66 +-
+ src/ops/__init__.py                      |    3 -
+ src/supervisor/app.py                    |  813 ++-----
+ src/supervisor/config.py                 |    7 -
+ src/supervisor/loop/__init__.py          |    6 -
+ src/supervisor/loop/arbiter.py           |   43 -
+ src/supervisor/loop/fixers.py            |  111 -
+ src/supervisor/loop/optimist.py          |   30 -
+ src/supervisor/loop/policy.py            |   88 -
+ src/supervisor/loop/policy_defaults.json |    9 -
+ src/supervisor/loop/skeptic.py           |   20 -
+ src/supervisor/loop/types.py             |   31 -
+ src/supervisor/models.py                 |   78 +-
+ src/supervisor/policy.py                 |    2 +-
+ src/supervisor/redact.py                 |   19 +-
+ src/supervisor/telegram_notify.py        |    2 +-
+ src/web/dashboard.py                     |    8 +-
+ tests/supervisor/test_loop_invariants.py |  243 --
+ tests/supervisor/test_loop_limits.py     |  217 --
+ tests/test_context_pack.py               |  104 +-
+ tests/test_context_pack_ops_health.py    |   70 +
+ tests/test_healthcheck_basic.py          |   19 +
+ tests/test_healthcheck_config.py         |   53 +-
+ tests/test_supervisor.py                 |    8 +-
+ 47 files changed, 5183 insertions(+), 4752 deletions(-)
 
 ## git diff origin/main..HEAD
-diff --git a/.vscode/tasks.json b/.vscode/tasks.json
-index 4b86157..31ea0c7 100644
---- a/.vscode/tasks.json
-+++ b/.vscode/tasks.json
-@@ -101,7 +101,7 @@
- 			"command": "/bin/zsh",
- 			"args": [
- 				"-lc",
--				"${workspaceFolder}/.venv/bin/python -m pytest -q > /tmp/pytest_out.txt 2>&1; echo \"EXIT:$?\"; tail -n 200 /tmp/pytest_out.txt"
-+				"cd \"${workspaceFolder}\" && .venv/bin/python -m pytest -q > /tmp/pytest_out.txt 2>&1; echo \"EXIT:$?\"; tail -n 200 /tmp/pytest_out.txt"
- 			],
- 			"isBackground": false,
- 			"group": "test"
-diff --git a/HEALTHCHECK.md b/HEALTHCHECK.md
-index 93e6324..25e7299 100644
---- a/HEALTHCHECK.md
-+++ b/HEALTHCHECK.md
-@@ -4,7 +4,131 @@ This document lists quick commands to verify that core parts of the system are w
- 
- ---
- 
--## Quick Smoke Tests
-+## Operational Health (watchdog-ready)
-+
-+This is the ops-grade health model used by automation and dashboards. It is designed for watchdogs and runtime guardrails.
-+
-+### Endpoints
-+
-+- `GET /api/ops/health/status` (cached)
-+- `POST /api/ops/health/run` (force refresh + cache)
-+
-+### Three Layers
-+
-+- **Liveness**: Core pipeline checks (config validity, Deribit connectivity, state builder).
-+- **Truth → Trust → Trade**:
-+- **Truth (facts)**: raw observations from the filesystem/stores (harvest presence + age, calibration last run, fidelity last run).
-+- **Trust (gates)**: normalized gate results with explicit `mode` (`off|warn|block`) and `status` (`PASS|WARN|FAIL`).
-+- **Trade (policy)**: aggregated `gate_overall` (`status|severity|can_trade`) used by dashboards and automation.
-+- **Decisions**: `overall_status`/`summary` are derived from `gate_overall` whenever gates are available; `checks_overall`/`checks_summary` are kept purely for diagnostics.
-+
-+### Thresholds & Policies
-+
-+- **Harvest freshness**:
-+  - OK: `age_minutes <= 60`
-+  - WARN: `60 < age_minutes <= 180`
-+  - FAIL: `age_minutes > 180` or missing files
-+- **Calibration freshness**:
-+  - OK: `last_calibration_at <= 36h` and applied
-+  - WARN: `36-72h` or `applied=False`
-+  - FAIL: `>72h`, missing bundle, or last run failed
-+- **Fidelity gate**:
-+  - TRUSTED: OK
-+  - WARNING: WARN (degraded)
-+  - UNTRUSTED: WARN by default; `HEALTH_STRICT_SYNTHETIC_GATE=1` escalates to FATAL + `can_trade=False`
-+  - Missing: WARN in research mode; FAIL in strict mode
-+
-+### Fidelity Report Store (Canonical)
-+
-+Ops-grade Synthetic Fidelity is persisted in a single canonical, file-based store:
-+
-+- Base dir: `data/fidelity_runs/` (override via `FIDELITY_DIR` or `FIDELITY_RUNS_DIR`)
-+- Per-run report: `data/fidelity_runs/<run_id>.json`
-+- Latest (full report): `data/fidelity_runs/latest.json`
-+- Latest per underlying (full report): `data/fidelity_runs/BTC/latest.json`, `data/fidelity_runs/ETH/latest.json`
-+
-+The latest *summary* for the dashboard endpoints is maintained separately:
-+
-+- `data/fidelity_runs/latest_summary.json`
-+- `data/fidelity_runs/index.jsonl`
-+
-+**Schema highlights** (fields health/gates rely on):
-+
-+- `run_id`, `created_at`, `underlying`
-+- `component_scores`: `underlying_returns`, `iv_surface_level`, `spot_iv_coupling`, `strategy_pnl_parity`
-+- `overall_score` (weighted combination), `gate_label` (`TRUSTED|WARNING|UNTRUSTED`)
-+- `thresholds`: `trusted_threshold`, `warn_threshold`, `min_coverage_ratio`
-+- `coverage.strategy_pnl_parity`: `valid_cases`, `total_cases`, `coverage_ratio_cases`, `min_trades_per_case`
-+
-+### Running the Ops-Grade Fidelity Suite
-+
-+This produces a canonical report in `data/fidelity_runs/` that is consumed by ops health and the unified gates.
-+
-+```bash
-+python -c "from src.fidelity.ops_runner import run_ops_fidelity_suite; run_ops_fidelity_suite(underlying='BTC', start_ts=1735689600, end_ts=1736121600)"
-+```
-+
-+> Strategy PnL parity uses Backtest Lab paired runs (`compare.run_synthetic_vs_live_pair`) and diffs (`diff.compute_diff_for_runs`).
-+
-+### Manual Ops Commands
-+
-+```bash
-+curl -s http://localhost:5000/api/ops/health/status
-+curl -s -X POST http://localhost:5000/api/ops/health/run
-+```
-+
-+> When `OPS_HEALTH_RUN_SECRET` is defined, add `-H "X-OPS-HEALTH-SECRET: $OPS_HEALTH_RUN_SECRET"` to the guarded POST so only authorized tooling can refresh the cache.
-+
-+### Guarding the Ops Health endpoint
-+
-+- `POST /api/ops/health/run` will reject requests without the matching `X-OPS-HEALTH-SECRET` header whenever `OPS_HEALTH_RUN_SECRET` is set. The handshake ensures operators cannot accidentally hammer Deribit / synthetic-data gates from a public dashboard.
-+- The dashboard’s “System Health” card is wired to `GET /api/ops/health/status`. When the cached status is missing (HTTP 404), the card displays “No cached health yet” and a call-to-action button. Clicking that button hits the guarded `/api/ops/health/run`, populates the cache, and re-renders the badge/summary once the data returns.
-+
-+### Gate Error Codes
-+
-+The unified gate framework uses standardized codes in each gate's `code` field:
-+
-+- Harvest: `NO_HARVESTED_FILES`, `HARVEST_RANGE_EMPTY`, `HARVEST_STALE`, `HARVEST_AGE_UNKNOWN`
-+- Fidelity: `FIDELITY_MISSING`, `FIDELITY_WARNING`, `FIDELITY_UNTRUSTED`, `FIDELITY_UNKNOWN`
-+- Calibration: `CALIBRATION_MISSING`, `CALIBRATION_FAILED`, `CALIBRATION_STALE`, `CALIBRATION_BLOCKED`, `CALIBRATION_AGE_UNKNOWN`
-+
-+---
-+
-+## Backtest Preflight (fail-fast)
-+
-+`POST /api/backtest/start` performs a **preflight** before spawning the backtest worker.
-+
-+- If the backtest is **historical** and uses `chain_mode=live_chain` (the default for historical), the system requires harvested snapshots under `data/live_deribit/*/*.parquet`.
-+- Preflight failures return a canonical error envelope:
-+
-+```json
-+{
-+  "ok": false,
-+  "error": {
-+    "code": "NO_HARVESTED_FILES",
-+    "message": "No harvested files available for requested date range.",
-+    "details": {
-+      "data_readiness": {
-+        "harvest_required": true,
-+        "harvest": {"available": false},
-+        "fidelity": {"available": false},
-+        "calibration": {"available": false}
-+      },
-+      "gates": [
-+        {"name": "harvest", "mode": "block", "status": "FAIL", "code": "NO_HARVESTED_FILES", "message": "No harvested files available."}
-+      ],
-+      "gate_overall": {"status": "FAIL", "severity": "FATAL", "can_trade": false},
-+      "effective_config": {"chain_mode": "live_chain", "is_historical": true}
-+    }
-+  }
-+}
-+```
-+
-+Preflight also enforces the optional fidelity gate (`FIDELITY_GATE_MODE=warn|block`) without spawning workers.
-+
-+## Smoke Tests (manual after changes)
-+
-+Smoke tests are manual and are not substitutes for the operational health checks above.
- 
- ### 1. Live Agent Dry-Run Test
- 
+diff --git a/.ci_bump b/.ci_bump
+deleted file mode 100644
+index ae63d90..0000000
+--- a/.ci_bump
++++ /dev/null
+@@ -1,3 +0,0 @@
+-
+-ci bump 2025-12-22T18:47:34Z
+-ci bump 2025-12-22T18:59:04Z
 diff --git a/Makefile b/Makefile
-new file mode 100644
-index 0000000..07d1860
---- /dev/null
+index 8f635ba..5bf1a75 100644
+--- a/Makefile
 +++ b/Makefile
-@@ -0,0 +1,14 @@
-+context-pack:
-+	python3 scripts/gen_repo_manifest.py
-+	bash scripts/gen_recent_diff.sh
-+	python3 scripts/gen_fidelity_latest_docs.py
-+
+@@ -1,15 +1,14 @@
+-.PHONY: context-pack context-pack-extras context-pack-all context-pack-push
+-
+-context-pack:
+-	./scripts/gen_repo_manifest.py
+-	./scripts/gen_repo_manifest_md.py
+-	./scripts/gen_recent_diff.sh
+-
+-context-pack-extras:
+-	./scripts/gen_roadmap_latest.sh
+-	./scripts/gen_test_summary_latest.sh
+-
+-context-pack-all: context-pack context-pack-extras
+-
+-context-pack-push: context-pack-all
+-	CONTEXT_PACK_PUSH_DIRECT=1 ./scripts/push_context_pack_to_drive.sh
 +extras:
 +	python3 scripts/gen_ops_health_latest.py
-+
-+context-pack-push: context-pack extras
 +	@if [ ! -f docs/TEST_SUMMARY_latest.txt ]; then \
 +		printf "%s\n%s\n" "$$(date -u +%Y-%m-%dT%H:%MZ)" "pytest summary unavailable" > docs/TEST_SUMMARY_latest.txt; \
 +	fi
 +	cp ROADMAP_BACKLOG.md docs/ROADMAP_BACKLOG_latest.md
 +	@echo "Updated docs/ROADMAP_BACKLOG_latest.md (upload handled externally)"
-diff --git a/POEM.md b/POEM.md
-new file mode 100644
-index 0000000..2744d6e
---- /dev/null
-+++ b/POEM.md
-@@ -0,0 +1,22 @@
-+# The Algorithm's Dance
 +
-+In circuits deep where logic flows,
-+A trader wakes, the market knows.
-+Each five-minute tick, a chance to see
-+What volatility might decree.
-+
-+The Greeks align in perfect form,
-+Delta, theta, ride the storm.
-+Black-Scholes whispers in the code,
-+Where synthetic prices find their road.
-+
-+Rule-based mind or LLM's grace,
-+Both paths converge at risk's gate.
-+Backtests replay what might have been,
-+The equity curve tells its tale.
-+
-+In testnet's safe and sandboxed realm,
-+The agent learns, the agent grows.
-+So here's to code that trades with care,
-+A faithful servant, always there.
++context-pack: extras
++	python3 scripts/gen_repo_manifest.py
++	bash scripts/gen_recent_diff.sh
++	python3 scripts/gen_fidelity_latest_docs.py
 +
-diff --git a/README.md b/README.md
-index dea428a..0752cb6 100644
---- a/README.md
-+++ b/README.md
-@@ -92,6 +92,47 @@ python agent_loop.py
- python -m backtest.env_simulator
- ```
++context-pack-push: context-pack
+diff --git a/POEM.md b/POEM.md
+index 2744d6e..ac8a4d2 100644
+--- a/POEM.md
++++ b/POEM.md
+@@ -20,3 +20,7 @@ The agent learns, the agent grows.
+ So here's to code that trades with care,
+ A faithful servant, always there.
  
-+## Context Pack & Drive Publishing
-+
-+This repo can generate a deterministic "context pack" under `docs/` for external consumers (e.g., a Google Drive folder used as LLM context).
-+
-+### Generate latest artifacts locally
-+
-+```bash
-+# Generates docs/REPO_MANIFEST.json, docs/RECENT_DIFF.md, and other "latest" artifacts
-+make context-pack-push
-+```
-+
-+### Fidelity “latest” artifacts
 +
-+If Fidelity has been run and the canonical store exists, the context-pack generator will also publish Fidelity reports into `docs/`:
 +
-+- `docs/FIDELITY_BTC_latest.json` / `docs/FIDELITY_BTC_latest.md`
-+  - copied from `data/fidelity_runs/BTC/latest/fidelity_report.json` and `.md`
-+- `docs/FIDELITY_ETH_latest.json` / `docs/FIDELITY_ETH_latest.md` (if present)
-+  - copied from `data/fidelity_runs/ETH/latest/fidelity_report.json` and `.md`
 +
-+Missing sources do **not** fail context-pack generation; the generator prints a warning and skips those files.
 +
-+You can also run the generator directly:
-+
-+```bash
-+python3 scripts/gen_fidelity_latest_docs.py
-+```
-+
-+### Upload to Google Drive (rclone)
-+
-+If you use `rclone` to publish the `docs/` “latest” artifacts to Drive, use:
-+
-+```bash
-+bash scripts/push_context_pack_to_drive.sh
-+```
-+
-+Notes:
-+- Requires `rclone` configured with a `gdrive:` remote.
-+- The target can be overridden with `CONTEXT_PACK_DRIVE_REMOTE`.
-+- To also upload a timestamped snapshot under `history/`, set `CONTEXT_PACK_UPLOAD_HISTORY=1`.
-+
- ## Project Structure
- 
- ```
 diff --git a/ROADMAP_BACKLOG.md b/ROADMAP_BACKLOG.md
-index 21bce55..131a69a 100644
+index 131a69a..74b0dcd 100644
 --- a/ROADMAP_BACKLOG.md
 +++ b/ROADMAP_BACKLOG.md
-@@ -811,3 +811,48 @@ The system supports three levels of automation, rolled out incrementally:
- 5. **Audit trail:** Every run, every LLM suggestion, and every parameter change is logged and traceable.
+@@ -815,6 +815,11 @@ The system supports three levels of automation, rolled out incrementally:
+ ## Changelog (auto)
+ - (entries appended newest-first)
  
- ---
-+
-+## Changelog (auto)
-+- (entries appended newest-first)
-+
-+- 2025-12-22T16:55Z [COPILOT] sha=fa5a78a
-+  - Summary: Refresh TEST_SUMMARY_latest.txt after full-suite run; Refresh context-pack latest artifacts prior to push
-+  - Tests: 822 passed, 5 skipped, 53 warnings in 335.18s (0:05:35)
-+  - Endpoints: none
-+  - Context-pack: uploaded (no)
-+- 2025-12-21T21:51Z [COPILOT] sha=3bc7e86
-+  - Summary: Document context-pack + Drive publishing (rclone) in README; Document Fidelity latest artifacts in docs/
-+  - Tests: not run (docs-only)
-+  - Endpoints: none
-+  - Context-pack: uploaded (no)
-+- 2025-12-21T20:38Z [CODEx] sha=3bc7e86546adf507d277ca9f65342246a7adc804
-+  - Summary: Gate_overall/summary drive decisions while checks stay diagnostic; - Fidelity gate check no longer toggles can_trade; - Harvest requirement + ops dashboard/tests align with truth→trust pipeline
-+  - Tests: ======================= 34 passed, 10 warnings in 7.82s ========================
-+  - Endpoints: /api/ops/health/run, /api/ops/health/status
-+  - Context-pack: uploaded (yes)
-+- 2025-12-21T20:34Z [COPILOT] sha=3bc7e86
-+  - Summary: Fix ROADMAP_BACKLOG duplicate Changelog (auto) header; Update changelog append script to target last section
-+  - Tests: not run (docs/script-only change)
++- 2025-12-22T19:11Z [COPILOT] sha=382ac49
++  - Summary: Ops health single-truth: gate_overall drives overall_status/summary when present; checks only block if can_trade=false; fail-closed on gate eval error when gate modes enabled
++  - Tests: 825 passed, 5 skipped (full suite)
 +  - Endpoints: none
 +  - Context-pack: uploaded (no)
-+- 2025-12-21T20:31Z [COPILOT] sha=3bc7e86
-+  - Summary: Add OPS_HEALTH_latest.json context-pack artifact generator (fake mode for tests); Wire ops health + test summary into context-pack-push; Make ops fidelity auditable: raw scores + clamping warnings + schema hard-fail
-+  - Tests: 5 passed, 4 warnings in 10.55s
-+  - Endpoints: none
-+  - Context-pack: uploaded (no)
-+- 2025-12-21T20:04Z [COPILOT] sha=3bc7e86
-+  - Summary: Make ops fidelity conservative: clamp scores + apply coverage penalty; Remove misleading gate_label parameters; missing_close always UNTRUSTED; Add tests for penalty + coverage schema
-+  - Tests: 28 passed, 5 warnings in 73.80s (0:01:13)
-+  - Endpoints: none
-+  - Context-pack: uploaded (no)
-+- 2025-12-21T19:22Z [CODEx] sha=3bc7e86546adf507d277ca9f65342246a7adc804
-+  - Summary: Gate-overall authoritative status/summary + checks_overall diagnostics; - Fidelity gate mode driven by env only; - Harvest required toggles + ops tests/UI updates
-+  - Tests: ======================= 31 passed, 10 warnings in 5.48s ========================
-+  - Endpoints: /api/ops/health/run, /api/ops/health/status
-+  - Context-pack: uploaded (yes)
-+
-+- 2025-12-21T19:10Z [COPILOT] sha=3bc7e86
-+  - Summary: Add roadmap changelog automation helper; Add pytest summary capture script (+ optional changelog append); Add context-pack-push target to refresh ROADMAP_BACKLOG_latest
-+  - Tests: 6 passed, 4 warnings in 1.50s
-+  - Endpoints: none
-+  - Context-pack: uploaded (no)
-diff --git a/agent_loop.py b/agent_loop.py
-index 673ea94..3e96631 100644
---- a/agent_loop.py
-+++ b/agent_loop.py
-@@ -36,6 +36,7 @@ from src.healthcheck import (
-     run_and_cache_healthcheck,
-     set_agent_paused_due_to_health,
-     is_agent_paused_due_to_health,
-+    get_cached_health_status,
- )
- from src.deribit.base_client import HealthSeverity
- 
-@@ -45,6 +46,18 @@ shutdown_requested = False
- last_health_recheck_time: float = 0
- 
- 
-+def _health_trading_allowed() -> tuple[bool, str]:
-+    """Return (allowed, reason) for trading based on cached health."""
-+    cached = get_cached_health_status()
-+    if cached is None:
-+        return False, "missing_cached_health"
-+    if cached.can_trade is False:
-+        severity = cached.worst_severity or "unknown"
-+        reason = cached.summary or "can_trade=False"
-+        return False, f"blocked_by_health can_trade=False severity={severity}: {reason}"
-+    return True, ""
-+
-+
- def signal_handler(signum: int, frame: object) -> None:
-     """Handle shutdown signals gracefully."""
-     global shutdown_requested
-@@ -302,6 +315,14 @@ def run_agent_loop_forever(
-             print(f"Iteration {iteration} - {datetime.utcnow().isoformat()}")
-             print(f"{'='*60}")
-             
-+            health_allowed, health_reason = _health_trading_allowed()
-+            if not health_allowed:
-+                if not is_agent_paused_due_to_health():
-+                    set_agent_paused_due_to_health(True)
-+                print(f"\n[HEALTH GUARD] blocked_by_health ({health_reason}). Skipping trading.")
-+                time.sleep(settings.loop_interval_sec)
-+                continue
-+
-             if is_agent_paused_due_to_health():
-                 print("\n[HEALTH GUARD] Agent paused due to health failure. Skipping trading.")
-                 print("[HEALTH GUARD] Will re-check health on next interval.")
-diff --git a/attached_assets/Pasted-ROLE-You-are-a-senior-product-minded-engineer-Update-th_1766256235012.txt b/attached_assets/Pasted-ROLE-You-are-a-senior-product-minded-engineer-Update-th_1766256235012.txt
-deleted file mode 100644
-index 41e79f8..0000000
---- a/attached_assets/Pasted-ROLE-You-are-a-senior-product-minded-engineer-Update-th_1766256235012.txt
-+++ /dev/null
-@@ -1,73 +0,0 @@
--ROLE
--You are a senior product-minded engineer. Update the repo’s roadmap file to reflect the new North Star discovered during Synthetic Fidelity + live replay work.
--
--FILE TO EDIT
--- ROADMAP_BACKLOG.md (if the repo uses a different name, locate the exact roadmap file; in this workspace it’s “ROADMAP_BACKLOG (2).md” content)
--
--CONTEXT YOU MUST PRESERVE
--The current roadmap already contains:
--- Phases 1–3 at the top
--- A strong section [E1.5] “Synthetic Fidelity Score + trading gate”
--- Recently Completed includes: synthetic regimes, live_deribit datasource, compare/diff modules, backtest lab UI, shared state_core, etc.
--
--You must keep the existing content, but REFRAME the plan so Fidelity Gate + measurement integrity are the organizing principle.
--
--GOALS
--1) Update the North Star / Phase plan to add a new Phase 0 (Truth Foundation + Fidelity Gate).
--2) Promote E1.5 from “one backlog item” into the prime directive, with enforcement language and clear exit criteria.
--3) Add a new P0 backlog section: “Measurement Integrity & Data Contracts” with concrete items.
--4) Ensure priorities are explicit: Truth → Trust → Trade, then scale strategies/bots, then SaaS.
--
--EDIT SPEC (DETAILED)
--
--A) Update the “Phases:” block near the top
--Replace the existing Phase list with:
--
--- Phase 0 – Truth Foundation + Fidelity Gate (new, must be complete before trusting backtests)
--- Phase 1 – One good covered-call bot on testnet with gate enforced (mostly done, but now “done” means TRUSTED)
--- Phase 2 – Strategy Packs (GregBot + others) with deterministic replay + parity
--- Phase 3 – Multi-bot supervisor + real historical data + production ops + heavy quant research
--
--Add a short “Definition of Done” bullet under the Phase list:
--- A strategy is “done” only if deterministic replay + audited PnL + passes Fidelity Gate.
--
--B) Add a new section near the top (after “Recently Completed” and before “A. Architecture & Design”)
--Title: “0. North Star: Truth → Trust → Trade”
--Include:
--- Why: synthetic backtests are untrusted unless fidelity gate is green
--- The gate labels: TRUSTED / WARNING / UNTRUSTED
--- Promotion ladder: synthetic → live replay → paper → testnet → mainnet
--- Hard rule: no auto-promotion; fidelity must pass
--
--C) Add a new P0 section (place it under “B. Persistence & Infrastructure” or create a new section “B0. Measurement Integrity & Data Contracts”)
--Items (P0):
--- Enforce premium units contract (option premiums are USD inside backtester; live_deribit premiums must be converted from underlying units)
--- Add invariant tests:
--  - drawdown sanity (non-negative, bounded in normal runs)
--  - PnL unit sanity (premium_usd vs underlying_price)
--- Document the contract in-doc: “units at each boundary” (harvester → exam → datasource → simulator)
--
--D) Update [E1.5] section
--Keep the existing content, but add:
--- “Enforcement” subsection:
--  - backtest UI must show gate label
--  - optimizer/strategy factory blocked unless TRUSTED (or tagged “exploratory-only”)
--  - live trading blocked unless latest calibration bundle is TRUSTED and not stale
--- “Exit criteria” subsection:
--  - min coverage threshold
--  - min strategy parity threshold for canonical strategies
--  - max allowed IV bucket MAE / vega-weighted MAE
--
--E) Add a small note under “G2 Backtest Lab enhancements” that fidelity badges must be shown
--(Do not build UI here; just make it explicit in roadmap that the backtest lab must surface gate status.)
--
--STYLE REQUIREMENTS
--- Keep the existing headings/IDs (A1, A2, E1.5, etc.) intact.
--- Add new sections without deleting old ones.
--- Be concise: roadmap is a control document, not an essay.
--
--ACCEPTANCE CRITERIA
--- Roadmap has Phase 0 added and clearly defines “done”
--- E1.5 explicitly states enforcement (gate blocks trust/trading)
--- New P0 “Measurement Integrity & Data Contracts” exists and is prioritized above strategy expansion
--- No loss of existing backlog items
-diff --git a/data/backtests/index.jsonl b/data/backtests/index.jsonl
-index 1b70a31..54d1939 100644
---- a/data/backtests/index.jsonl
-+++ b/data/backtests/index.jsonl
-@@ -1,3 +1,21 @@
-+{"run_id": "2025-12-20T18-00-49Z_BTC_8696f53d", "created_at": "2025-12-20T18:00:49.037207+00:00", "underlying": "BTC", "start_date": "2025-12-07", "end_date": "2025-12-13", "status": "finished", "primary_exit_style": "tp_and_roll", "net_profit_pct": 17.8141689, "max_drawdown_pct": 69.74500725391692, "sharpe_ratio": 0.0, "num_trades": 8}
-+{"run_id": "2025-12-20T18-00-44Z_BTC_6f0d5259", "created_at": "2025-12-20T18:00:44.828734+00:00", "underlying": "BTC", "start_date": "2025-12-07", "end_date": "2025-12-13", "status": "finished", "primary_exit_style": "tp_and_roll", "net_profit_pct": 15.520456718055934, "max_drawdown_pct": 75.75119335301032, "sharpe_ratio": 0.0, "num_trades": 8}
-+{"run_id": "2025-12-20T18-00-42Z_BTC_aca321ad", "created_at": "2025-12-20T18:00:42.294581+00:00", "underlying": "BTC", "start_date": "2025-12-07", "end_date": "2025-12-13", "status": "finished", "primary_exit_style": "hold_to_expiry", "net_profit_pct": -96.8166285098635, "max_drawdown_pct": 732.1046240677935, "sharpe_ratio": 0.0, "num_trades": 8}
-+{"run_id": "2025-12-20T18-00-38Z_BTC_872794f5", "created_at": "2025-12-20T18:00:38.743247+00:00", "underlying": "BTC", "start_date": "2025-12-07", "end_date": "2025-12-13", "status": "finished", "primary_exit_style": "hold_to_expiry", "net_profit_pct": -91.98114545064861, "max_drawdown_pct": 709.7279509351115, "sharpe_ratio": 0.0, "num_trades": 8}
-+{"run_id": "2025-12-20T17-48-26Z_BTC_9ffec9e8", "created_at": "2025-12-20T17:48:26.991459+00:00", "underlying": "BTC", "start_date": "2025-12-10", "end_date": "2025-12-13", "status": "finished", "primary_exit_style": "tp_and_roll", "net_profit_pct": 57.59303349999999, "max_drawdown_pct": 0.0, "sharpe_ratio": 0.0, "num_trades": 20}
-+{"run_id": "2025-12-20T17-48-23Z_BTC_bd092e41", "created_at": "2025-12-20T17:48:23.656234+00:00", "underlying": "BTC", "start_date": "2025-12-10", "end_date": "2025-12-13", "status": "finished", "primary_exit_style": "tp_and_roll", "net_profit_pct": 62.782024643397435, "max_drawdown_pct": 0.0, "sharpe_ratio": 0.0, "num_trades": 20}
-+{"run_id": "2025-12-20T17-48-20Z_BTC_1c7aece6", "created_at": "2025-12-20T17:48:20.169591+00:00", "underlying": "BTC", "start_date": "2025-12-10", "end_date": "2025-12-13", "status": "finished", "primary_exit_style": "hold_to_expiry", "net_profit_pct": -79.67056649999998, "max_drawdown_pct": 10926.100180886597, "sharpe_ratio": 0.0, "num_trades": 20}
-+{"run_id": "2025-12-20T17-48-16Z_BTC_a04db4cb", "created_at": "2025-12-20T17:48:16.050770+00:00", "underlying": "BTC", "start_date": "2025-12-10", "end_date": "2025-12-13", "status": "finished", "primary_exit_style": "hold_to_expiry", "net_profit_pct": -74.48157535660285, "max_drawdown_pct": 5301.185246651486, "sharpe_ratio": 0.0, "num_trades": 20}
-+{"run_id": "2025-12-20T17-46-15Z_BTC_7a9fcec4", "created_at": "2025-12-20T17:46:15.098090+00:00", "underlying": "BTC", "start_date": "2025-12-10", "end_date": "2025-12-13", "status": "finished", "primary_exit_style": "tp_and_roll", "net_profit_pct": 0.0, "max_drawdown_pct": 0.0, "sharpe_ratio": 0.0, "num_trades": 0}
-+{"run_id": "2025-12-20T17-46-11Z_BTC_61c04c6b", "created_at": "2025-12-20T17:46:11.806403+00:00", "underlying": "BTC", "start_date": "2025-12-10", "end_date": "2025-12-13", "status": "finished", "primary_exit_style": "tp_and_roll", "net_profit_pct": 10.129996053522591, "max_drawdown_pct": 20.785119319814665, "sharpe_ratio": 0.0, "num_trades": 12}
-+{"run_id": "2025-12-20T17-46-08Z_BTC_17c49061", "created_at": "2025-12-20T17:46:08.481364+00:00", "underlying": "BTC", "start_date": "2025-12-10", "end_date": "2025-12-13", "status": "finished", "primary_exit_style": "hold_to_expiry", "net_profit_pct": 0.0, "max_drawdown_pct": 0.0, "sharpe_ratio": 0.0, "num_trades": 0}
-+{"run_id": "2025-12-20T17-46-04Z_BTC_d9fdfdc1", "created_at": "2025-12-20T17:46:04.487702+00:00", "underlying": "BTC", "start_date": "2025-12-10", "end_date": "2025-12-13", "status": "finished", "primary_exit_style": "hold_to_expiry", "net_profit_pct": -11.156403946477367, "max_drawdown_pct": 846.6285195195594, "sharpe_ratio": 0.0, "num_trades": 12}
-+{"run_id": "2025-12-20T17-44-18Z_BTC_3c3acec3", "created_at": "2025-12-20T17:44:18.961916+00:00", "underlying": "BTC", "start_date": "2025-12-10", "end_date": "2025-12-13", "status": "failed", "primary_exit_style": "tp_and_roll", "net_profit_pct": 0.0, "max_drawdown_pct": 0.0, "sharpe_ratio": 0.0, "num_trades": 0, "error": "[build_historical_state] Both live_chain and synthetic_grid returned empty candidates at 2025-12-10 19:00:00+00:00 for BTC. sigma=0.1000, spot=92938.54, DTE range=[1, 21]"}
-+{"run_id": "2025-12-20T17-44-16Z_BTC_5f1c7ed7", "created_at": "2025-12-20T17:44:16.034027+00:00", "underlying": "BTC", "start_date": "2025-12-10", "end_date": "2025-12-13", "status": "failed", "primary_exit_style": "hold_to_expiry", "net_profit_pct": 0.0, "max_drawdown_pct": 0.0, "sharpe_ratio": 0.0, "num_trades": 0, "error": "[build_historical_state] Both live_chain and synthetic_grid returned empty candidates at 2025-12-10 19:00:00+00:00 for BTC. sigma=0.1000, spot=92938.54, DTE range=[1, 21]"}
-+{"run_id": "2025-12-20T17-35-40Z_BTC_7f53a08a", "created_at": "2025-12-20T17:35:40.990440+00:00", "underlying": "BTC", "start_date": "2025-12-10", "end_date": "2025-12-13", "status": "finished", "primary_exit_style": "tp_and_roll", "net_profit_pct": 0, "max_drawdown_pct": 0.0, "sharpe_ratio": 0, "num_trades": 0}
-+{"run_id": "2025-12-20T17-35-40Z_BTC_f98837c0", "created_at": "2025-12-20T17:35:40.979393+00:00", "underlying": "BTC", "start_date": "2025-12-10", "end_date": "2025-12-13", "status": "finished", "primary_exit_style": "tp_and_roll", "net_profit_pct": 0, "max_drawdown_pct": 0.0, "sharpe_ratio": 0, "num_trades": 0}
-+{"run_id": "2025-12-20T17-35-40Z_BTC_149b1f38", "created_at": "2025-12-20T17:35:40.967545+00:00", "underlying": "BTC", "start_date": "2025-12-10", "end_date": "2025-12-13", "status": "finished", "primary_exit_style": "hold_to_expiry", "net_profit_pct": 0, "max_drawdown_pct": 0.0, "sharpe_ratio": 0, "num_trades": 0}
-+{"run_id": "2025-12-20T17-35-40Z_BTC_c7f1ed8c", "created_at": "2025-12-20T17:35:40.949641+00:00", "underlying": "BTC", "start_date": "2025-12-10", "end_date": "2025-12-13", "status": "finished", "primary_exit_style": "hold_to_expiry", "net_profit_pct": 0, "max_drawdown_pct": 0.0, "sharpe_ratio": 0, "num_trades": 0}
- {"run_id": "2025-12-18T22-57-20Z_BTC_4b2651a4", "created_at": "2025-12-18T22:57:20.501665", "underlying": "BTC", "start_date": "2024-01-01T00:00:00+00:00", "end_date": "2024-01-07T00:00:00+00:00", "status": "finished", "primary_exit_style": "hold_to_expiry", "net_profit_pct": 0.0, "max_drawdown_pct": 0.0, "sharpe_ratio": 0.0, "num_trades": 0}
- {"run_id": "2025-12-18T21-07-07Z_BTC_506dfd07", "created_at": "2025-12-18T21:07:07.366533", "underlying": "BTC", "start_date": "2024-01-01T00:00:00+00:00", "end_date": "2024-01-07T00:00:00+00:00", "status": "finished", "primary_exit_style": "hold_to_expiry", "net_profit_pct": 0.0, "max_drawdown_pct": 0.0, "sharpe_ratio": 0.0, "num_trades": 0}
- {"run_id": "2025-12-18T19-08-51Z_BTC_5fdeffcb", "created_at": "2025-12-18T19:08:51.281554", "underlying": "BTC", "start_date": "2024-01-01T00:00:00+00:00", "end_date": "2024-01-07T00:00:00+00:00", "status": "finished", "primary_exit_style": "hold_to_expiry", "net_profit_pct": 0.0, "max_drawdown_pct": 0.0, "sharpe_ratio": 0.0, "num_trades": 0}
-diff --git a/data/fidelity_runs/BTC/history/20251219_234717/fidelity_report.json b/data/fidelity_runs/BTC/history/20251219_234717/fidelity_report.json
-deleted file mode 100644
-index 135e9e4..0000000
---- a/data/fidelity_runs/BTC/history/20251219_234717/fidelity_report.json
-+++ /dev/null
-@@ -1,131 +0,0 @@
--{
--  "component_scores": {
--    "strategy_pnl_parity": 100.0,
--    "underlying_returns": 100.0
--  },
--  "gate": "TRUSTED",
--  "market_live_meta": {
--    "ds_class": "LiveDeribitDataSource",
--    "margin_type": "linear",
--    "settlement_ccy": "USDC",
--    "type": "live_replay",
--    "underlying": "BTC"
--  },
--  "market_synth_meta": {
--    "cfg_class": "CallSimulationConfig",
--    "type": "synthetic_replay",
--    "underlying": "BTC"
+ - 2025-12-22T16:55Z [COPILOT] sha=fa5a78a
+   - Summary: Refresh TEST_SUMMARY_latest.txt after full-suite run; Refresh context-pack latest artifacts prior to push
+   - Tests: 822 passed, 5 skipped, 53 warnings in 335.18s (0:05:35)
+diff --git a/docs/FIDELITY_BTC_latest.json b/docs/FIDELITY_BTC_latest.json
+index ff44c7f..135e9e4 100644
+--- a/docs/FIDELITY_BTC_latest.json
++++ b/docs/FIDELITY_BTC_latest.json
+@@ -1,118 +1,9 @@
+ {
+   "component_scores": {
+-    "iv_surface_level": 100.0,
+-    "spot_iv_coupling": 100.0,
+-    "strategy_pnl_parity": 100.00000000000001,
++    "strategy_pnl_parity": 100.0,
+     "underlying_returns": 100.0
+   },
+-  "component_status": {
+-    "iv_surface_level": "not_available",
+-    "spot_iv_coupling": "not_available",
+-    "strategy_pnl_parity": "not_available",
+-    "underlying_returns": "not_available"
 -  },
--  "overall_score": 100.0,
--  "run_id": "20251219_234717",
--  "strategy_parity": {
--    "decision_times": [
--      "2025-12-07T00:00:00+00:00",
--      "2025-12-08T00:00:00+00:00",
--      "2025-12-09T00:00:00+00:00",
--      "2025-12-10T00:00:00+00:00",
--      "2025-12-11T00:00:00+00:00",
--      "2025-12-12T00:00:00+00:00",
--      "2025-12-13T00:00:00+00:00",
--      "2025-12-14T00:00:00+00:00",
--      "2025-12-15T00:00:00+00:00",
--      "2025-12-16T00:00:00+00:00",
--      "2025-12-17T00:00:00+00:00",
--      "2025-12-18T00:00:00+00:00",
--      "2025-12-19T00:00:00+00:00"
--    ],
--    "strategies": [
--      {
--        "live": {
--          "notes": "P0 placeholder (execution not implemented)",
--          "num_trades": 0,
--          "spot_first": 0.0,
--          "spot_last": 0.0
--        },
--        "name": "covered_call",
--        "synthetic": {
--          "notes": "P0 placeholder (execution not implemented)",
--          "num_trades": 0,
--          "spot_first": 0.0,
--          "spot_last": 85758.41
--        }
--      },
--      {
--        "live": {
--          "notes": "P0 placeholder (execution not implemented)",
--          "num_trades": 0,
--          "spot_first": 0.0,
--          "spot_last": 0.0
--        },
--        "name": "cash_secured_put",
--        "synthetic": {
--          "notes": "P0 placeholder (execution not implemented)",
--          "num_trades": 0,
--          "spot_first": 0.0,
--          "spot_last": 85758.41
--        }
+-  "components": {
+-    "iv_surface_level": {
+-      "meta": {
+-        "coverage": 0.0,
+-        "mae": null,
+-        "mae_by_bucket": {}
 -      },
--      {
--        "live": {
--          "notes": "P0 placeholder (execution not implemented)",
--          "num_trades": 0,
--          "spot_first": 0.0,
--          "spot_last": 0.0
--        },
--        "name": "short_strangle",
--        "synthetic": {
--          "notes": "P0 placeholder (execution not implemented)",
--          "num_trades": 0,
--          "spot_first": 0.0,
--          "spot_last": 85758.41
+-      "metrics": {
+-        "iv_bucket_mae": {
+-          "error": 0.0,
+-          "k": 1.0,
+-          "tolerance": 0.05,
+-          "weight": 1.0
 -        }
 -      },
--      {
--        "live": {
--          "notes": "P0 placeholder (execution not implemented)",
--          "num_trades": 0,
--          "spot_first": 0.0,
--          "spot_last": 0.0
--        },
--        "name": "put_spread_credit",
--        "synthetic": {
--          "notes": "P0 placeholder (execution not implemented)",
--          "num_trades": 0,
--          "spot_first": 0.0,
--          "spot_last": 85758.41
--        }
--      },
--      {
--        "live": {
--          "notes": "P0 placeholder (execution not implemented)",
--          "num_trades": 0,
--          "spot_first": 0.0,
--          "spot_last": 0.0
--        },
--        "name": "call_spread_debit",
--        "synthetic": {
--          "notes": "P0 placeholder (execution not implemented)",
--          "num_trades": 0,
--          "spot_first": 0.0,
--          "spot_last": 85758.41
--        }
--      },
--      {
--        "live": {
--          "notes": "P0 placeholder (execution not implemented)",
--          "num_trades": 0,
--          "spot_first": 0.0,
--          "spot_last": 0.0
--        },
--        "name": "calendar",
--        "synthetic": {
--          "notes": "P0 placeholder (execution not implemented)",
--          "num_trades": 0,
--          "spot_first": 0.0,
--          "spot_last": 85758.41
--        }
--      }
--    ]
--  },
--  "timestamp": "2025-12-19T23:47:17.593360+00:00"
--}
-\ No newline at end of file
-diff --git a/data/fidelity_runs/BTC/history/20251219_234717/fidelity_report.md b/data/fidelity_runs/BTC/history/20251219_234717/fidelity_report.md
-deleted file mode 100644
-index 68f3de9..0000000
---- a/data/fidelity_runs/BTC/history/20251219_234717/fidelity_report.md
-+++ /dev/null
-@@ -1,147 +0,0 @@
--# Synthetic Fidelity Report
--
--- Run ID: 20251219_234717
--- Timestamp (UTC): 2025-12-19T23:47:17.593360+00:00
--- Gate: **TRUSTED**
--
--## Scores
--
--- Overall: **100.0**
--- strategy_pnl_parity: 100.0
--- underlying_returns: 100.0
--
--## Market Meta
--
--### Live
--```json
--{
--  "ds_class": "LiveDeribitDataSource",
--  "margin_type": "linear",
--  "settlement_ccy": "USDC",
--  "type": "live_replay",
--  "underlying": "BTC"
--}
--```
--
--### Synthetic
--```json
--{
--  "cfg_class": "CallSimulationConfig",
--  "type": "synthetic_replay",
--  "underlying": "BTC"
--}
--```
--
--## Strategy Parity (P0 placeholder)
--
--```json
--{
--  "decision_times": [
--    "2025-12-07T00:00:00+00:00",
--    "2025-12-08T00:00:00+00:00",
--    "2025-12-09T00:00:00+00:00",
--    "2025-12-10T00:00:00+00:00",
--    "2025-12-11T00:00:00+00:00",
--    "2025-12-12T00:00:00+00:00",
--    "2025-12-13T00:00:00+00:00",
--    "2025-12-14T00:00:00+00:00",
--    "2025-12-15T00:00:00+00:00",
--    "2025-12-16T00:00:00+00:00",
--    "2025-12-17T00:00:00+00:00",
--    "2025-12-18T00:00:00+00:00",
--    "2025-12-19T00:00:00+00:00"
--  ],
--  "strategies": [
--    {
--      "live": {
--        "notes": "P0 placeholder (execution not implemented)",
--        "num_trades": 0,
--        "spot_first": 0.0,
--        "spot_last": 0.0
--      },
--      "name": "covered_call",
--      "synthetic": {
--        "notes": "P0 placeholder (execution not implemented)",
--        "num_trades": 0,
--        "spot_first": 0.0,
--        "spot_last": 85758.41
--      }
--    },
--    {
--      "live": {
--        "notes": "P0 placeholder (execution not implemented)",
--        "num_trades": 0,
--        "spot_first": 0.0,
--        "spot_last": 0.0
--      },
--      "name": "cash_secured_put",
--      "synthetic": {
--        "notes": "P0 placeholder (execution not implemented)",
--        "num_trades": 0,
--        "spot_first": 0.0,
--        "spot_last": 85758.41
--      }
--    },
--    {
--      "live": {
--        "notes": "P0 placeholder (execution not implemented)",
--        "num_trades": 0,
--        "spot_first": 0.0,
--        "spot_last": 0.0
--      },
--      "name": "short_strangle",
--      "synthetic": {
--        "notes": "P0 placeholder (execution not implemented)",
--        "num_trades": 0,
--        "spot_first": 0.0,
--        "spot_last": 85758.41
--      }
+-      "status": "not_available",
+-      "weight": 0.3
 -    },
--    {
--      "live": {
--        "notes": "P0 placeholder (execution not implemented)",
--        "num_trades": 0,
--        "spot_first": 0.0,
--        "spot_last": 0.0
+-    "spot_iv_coupling": {
+-      "meta": {
+-        "corr_live": null,
+-        "corr_synth": null
 -      },
--      "name": "put_spread_credit",
--      "synthetic": {
--        "notes": "P0 placeholder (execution not implemented)",
--        "num_trades": 0,
--        "spot_first": 0.0,
--        "spot_last": 85758.41
--      }
--    },
--    {
--      "live": {
--        "notes": "P0 placeholder (execution not implemented)",
--        "num_trades": 0,
--        "spot_first": 0.0,
--        "spot_last": 0.0
+-      "metrics": {
+-        "corr_spot_div_diff": {
+-          "error": 0.0,
+-          "k": 1.0,
+-          "tolerance": 0.3,
+-          "weight": 1.0
+-        }
 -      },
--      "name": "call_spread_debit",
--      "synthetic": {
--        "notes": "P0 placeholder (execution not implemented)",
--        "num_trades": 0,
--        "spot_first": 0.0,
--        "spot_last": 85758.41
--      }
+-      "status": "not_available",
+-      "weight": 0.2
 -    },
--    {
--      "live": {
--        "notes": "P0 placeholder (execution not implemented)",
--        "num_trades": 0,
--        "spot_first": 0.0,
--        "spot_last": 0.0
+-    "strategy_pnl_parity": {
+-      "meta": {
+-        "n_strategies": 6
 -      },
--      "name": "calendar",
--      "synthetic": {
--        "notes": "P0 placeholder (execution not implemented)",
--        "num_trades": 0,
--        "spot_first": 0.0,

TRUNCATED
