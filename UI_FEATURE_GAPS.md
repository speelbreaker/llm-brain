# UI Feature Gaps Report

This document identifies which important features and configuration options exist **only in code** versus what is currently exposed in the web dashboard. Use it to prioritize future UI enhancements that would give operators better visibility and control without touching environment variables or code.

---

## High-Level Summary

| Category | Count |
|----------|-------|
| **Total features inspected** | 38 |
| **Full UI (control + visibility)** | 8 |
| **Partial UI (visibility only OR control only)** | 11 |
| **Code only (no UI)** | 19 |

---

## Detailed Feature Table

### Risk & Safety Controls

| Feature | Category | Code Location(s) | UI Control? | UI Display? | Proposed UI Element | Suggested Placement | Notes |
|---------|----------|------------------|-------------|-------------|---------------------|---------------------|-------|
| **Kill switch** | Risk | `src/config.Settings.kill_switch_enabled`, `src/risk_engine.check_action_allowed()` | no | yes (System Health tab shows status) | Toggle switch: "Global Kill Switch" with confirmation modal | System Health card, top of "Risk Limits" section | Currently read-only display. Adding toggle would allow emergency halt without env var change. |
| **Daily drawdown limit** | Risk | `src/config.Settings.daily_drawdown_limit_pct`, `src/risk_engine._check_daily_drawdown_limit()` | no | yes (System Health tab shows value) | Number input: "Daily Drawdown Limit %" (0 = disabled) | System Health card, Risk Limits section | Value displayed but not editable. Runtime change would let operators tighten/relax guard. |
| **Max margin used %** | Risk | `src/config.Settings.max_margin_used_pct`, `src/risk_engine.check_action_allowed()` | no | yes (System Health tab) | Number input: "Max Margin Usage %" | System Health card, Risk Limits section | Displayed in risk status line; adding control would allow dynamic adjustment. |
| **Max net delta** | Risk | `src/config.Settings.max_net_delta_abs`, `src/risk_engine.check_action_allowed()` | no | yes (System Health tab) | Number input: "Max Net Delta" | System Health card, Risk Limits section | Displayed; no control to modify at runtime. |
| **Max expiry exposure** | Risk | `src/config.Settings.max_expiry_exposure`, `src/risk_engine.check_action_allowed()` | no | no | Number input: "Max Per-Expiry Exposure" (BTC/ETH) | System Health card, Risk Limits section | Not displayed or controllable. Important for concentration risk. |
| **Position reconcile action** | Risk | `src/config.Settings.position_reconcile_action` ("halt" / "auto_heal") | no | no | Dropdown: "On Position Mismatch" (Halt / Auto-Heal) | System Health card, Position Reconciliation section | Hidden config that determines recovery behavior. |
| **Position reconcile on startup** | Risk | `src/config.Settings.position_reconcile_on_startup` | no | no | Checkbox: "Run reconciliation on startup" | System Health card, Position Reconciliation section | Currently always on if configured. |
| **Position reconcile on each loop** | Risk | `src/config.Settings.position_reconcile_on_each_loop` | no | no | Checkbox: "Run reconciliation on each loop" | System Health card, Position Reconciliation section | May want to disable temporarily for debugging. |
| **Position reconcile tolerance (USD)** | Risk | `src/config.Settings.position_reconcile_tolerance_usd` | no | no | Number input: "Mismatch Tolerance (USD)" | System Health card, Position Reconciliation section | Fine-tuning parameter for rounding diffs. |

### LLM & Decision Mode

| Feature | Category | Code Location(s) | UI Control? | UI Display? | Proposed UI Element | Suggested Placement | Notes |
|---------|----------|------------------|-------------|-------------|---------------------|---------------------|-------|
| **LLM enabled** | LLM/Decisions | `src/config.Settings.llm_enabled` | no | yes (System Health tab, header badges) | Toggle switch: "Enable LLM Decisions" | System Health card, LLM section header | Displayed but not runtime-toggleable. |
| **Decision mode** | LLM/Decisions | `src/config.Settings.decision_mode` ("rule_only", "llm_only", "hybrid_shadow") | no | yes (System Health tab shows mode) | Dropdown: "Decision Mode" (Rule Only / LLM Only / Hybrid Shadow) | System Health card, LLM section | Major operational control currently env-only. |
| **LLM shadow enabled** | LLM/Decisions | `src/config.Settings.llm_shadow_enabled` | no | yes (System Health tab) | Checkbox: "Run LLM in shadow mode" | System Health card, LLM section | Visible but not toggleable. |
| **LLM validation strict** | LLM/Decisions | `src/config.Settings.llm_validation_strict` | no | yes (System Health tab) | Checkbox: "Strict LLM validation" | System Health card, LLM section | Visible but not toggleable. |
| **LLM model name** | LLM/Decisions | `src/config.Settings.llm_model_name` | no | no | Text input or dropdown: "LLM Model" | System Health card, LLM section (advanced) | Hidden; could allow switching between models. |
| **LLM timeout** | LLM/Decisions | `src/config.Settings.llm_timeout_seconds` | no | no | Number input: "LLM Timeout (sec)" | System Health card, LLM section (advanced) | Hidden parameter. |

### Training Mode

| Feature | Category | Code Location(s) | UI Control? | UI Display? | Proposed UI Element | Suggested Placement | Notes |
|---------|----------|------------------|-------------|-------------|---------------------|---------------------|-------|
| **Training mode toggle** | Training | `src/config.Settings.training_mode`, `/api/training/toggle` | yes | yes | Toggle switch (already exists in header) | Header bar (already present) | Fully implemented. |
| **Training profile mode** | Training | `src/config.Settings.training_profile_mode` ("single", "ladder") | no | no | Dropdown: "Training Profile" (Single / Ladder) | Live Agent tab, Training panel OR System Health | Affects how many positions are opened. |
| **Training strategies list** | Training | `src/config.Settings.training_strategies` | no | yes (badge shows strategies) | Multi-select or text: "Training Strategies" | Live Agent tab, Training panel | Visible in badge but not editable. |
| **Max calls per underlying (training)** | Training | `src/config.Settings.max_calls_per_underlying_training` | no | no | Number input: "Max Calls/Underlying (Training)" | System Health or Training panel | Hidden; controls ladder depth. |
| **Max calls per expiry (training)** | Training | `src/config.Settings.training_max_calls_per_expiry` | no | no | Number input: "Max Calls/Expiry (Training)" | System Health or Training panel | Hidden; controls per-expiry cap. |
| **Save training data** | Training | `src/config.Settings.save_training_data` | no | no | Checkbox: "Save Training Data CSVs" | Backtest panel or Training panel | Important for ML data collection. |

### Agent Configuration

| Feature | Category | Code Location(s) | UI Control? | UI Display? | Proposed UI Element | Suggested Placement | Notes |
|---------|----------|------------------|-------------|-------------|---------------------|---------------------|-------|
| **Operating mode** | Config | `src/config.Settings.mode` ("research", "production") | no | yes (header badge) | Dropdown: "Operating Mode" (Research / Production) | Header bar or System Health | Displayed but env-only change. |
| **Deribit environment** | Config | `src/config.Settings.deribit_env` ("testnet", "mainnet") | no | yes (System Health tab) | Read-only indicator (intentionally no control) | System Health card, prominent position | Changing this is dangerous; display-only is correct. |
| **Dry run mode** | Config | `src/config.Settings.dry_run` | no | yes (header badge) | Toggle switch: "Dry Run Mode" | Header bar or Live Agent tab | Displayed but not toggleable. Critical safety control. |
| **Loop interval** | Config | `src/config.Settings.loop_interval_sec` | no | no | Number input: "Loop Interval (seconds)" | System Health card or Advanced Settings | Hidden; affects decision frequency. |
| **Default order size** | Config | `src/config.Settings.default_order_size` | no | no | Number input: "Default Order Size (BTC/ETH)" | System Health or Live Agent | Hidden; affects all trades. |
| **Underlyings list** | Config | `src/config.Settings.underlyings` | no | no | Multi-select: "Active Underlyings" (BTC, ETH) | System Health or Live Agent | Usually ["BTC", "ETH"]. |
| **Option margin type** | Config | `src/config.Settings.option_margin_type` ("linear", "inverse") | no | no | Dropdown: "Margin Type" (Linear / Inverse) | Backtest panel or System Health | Important for Deribit product selection. |

### Strategy Thresholds

| Feature | Category | Code Location(s) | UI Control? | UI Display? | Proposed UI Element | Suggested Placement | Notes |
|---------|----------|------------------|-------------|-------------|---------------------|---------------------|-------|
| **IVRV minimum** | Strategy | `src/config.Settings.ivrv_min` / `research_ivrv_min` | no | no | Number input: "Min IV/RV Ratio" | Live Agent, Strategy Rules panel | Core filter threshold. |
| **Delta range** | Strategy | `src/config.Settings.delta_min` / `delta_max` + research variants | no | no | Range slider: "Delta Range (min - max)" | Live Agent, Strategy Rules panel | Core filter threshold. |
| **DTE range** | Strategy | `src/config.Settings.dte_min` / `dte_max` + research variants | no | no | Range slider: "DTE Range (min - max)" | Live Agent, Strategy Rules panel | Core filter threshold. |
| **Minimum premium (USD)** | Strategy | `src/config.Settings.premium_min_usd` | no | no | Number input: "Min Premium (USD)" | Live Agent, Strategy Rules panel | Filters out small premiums. |
| **Exploration probability** | Strategy | `src/config.Settings.explore_prob` | no | yes (header badge shows %) | Slider: "Explore Probability %" | Live Agent tab, Strategy panel | Visible but not adjustable. |
| **Explore top-k** | Strategy | `src/config.Settings.explore_top_k` | no | no | Number input: "Explore Top-K Candidates" | Live Agent tab, Strategy panel (advanced) | Hidden; affects exploration breadth. |

### Synthetic Pricing / Backtest

| Feature | Category | Code Location(s) | UI Control? | UI Display? | Proposed UI Element | Suggested Placement | Notes |
|---------|----------|------------------|-------------|-------------|---------------------|---------------------|-------|
| **Synthetic IV multiplier** | Backtest | `src/config.Settings.synthetic_iv_multiplier` | yes (calibration tab) | yes (calibration tab) | Number input (already exists in Calibration tab) | Calibration tab | Fully implemented. |
| **Synthetic skew enabled** | Backtest | `src/config.Settings.synthetic_skew_enabled` | no | no | Checkbox: "Enable Synthetic Skew" | Calibration tab or Backtest panel | Hidden; affects pricing accuracy. |
| **Synthetic skew DTE range** | Backtest | `src/config.Settings.synthetic_skew_min_dte`, `synthetic_skew_max_dte` | no | no | Range inputs: "Skew DTE Range" | Calibration tab (advanced) | Hidden fine-tuning. |

### Diagnostics

| Feature | Category | Code Location(s) | UI Control? | UI Display? | Proposed UI Element | Suggested Placement | Notes |
|---------|----------|------------------|-------------|-------------|---------------------|---------------------|-------|
| **Full agent healthcheck** | Diagnostics | `src/healthcheck.run_agent_healthcheck()`, `/api/agent_healthcheck` | yes | yes | Button: "Run Full Agent Healthcheck" (already exists) | System Health tab | Fully implemented with results display. |
| **Test LLM decision** | Diagnostics | `/api/test_llm_decision` | yes | yes | Button: "Test LLM Decision Pipeline" (already exists) | System Health tab | Fully implemented. |
| **Run reconciliation** | Diagnostics | `src/reconciliation.run_reconciliation_once()`, `/api/reconcile_positions` | yes | yes | Button: "Run Reconciliation Now" (already exists) | System Health tab | Fully implemented. |
| **Test kill switch** | Diagnostics | `/api/test_kill_switch` | yes | yes | Button: "Test Risk Checks / Kill Switch" (already exists) | System Health tab | Fully implemented. |

---

## Summary by Priority

### High Priority (Safety & Core Operations)

| Feature | Current State | Recommended UI |
|---------|---------------|----------------|
| Kill switch toggle | Display only | Add toggle switch with confirmation |
| Daily drawdown limit | Display only | Add editable input |
| Decision mode selector | Display only | Add dropdown selector |
| Dry run mode toggle | Display only | Add toggle switch |
| Position reconcile action | Hidden | Add dropdown (halt/auto_heal) |

### Medium Priority (Operational Flexibility)

| Feature | Current State | Recommended UI |
|---------|---------------|----------------|
| LLM enabled toggle | Display only | Add toggle switch |
| Max margin/delta limits | Display only | Add editable inputs |
| Training profile mode | Hidden | Add dropdown selector |
| Strategy thresholds (IVRV, delta, DTE) | Hidden | Add filter controls panel |
| Exploration probability | Display only | Add slider |

### Low Priority (Advanced/Rare Use)

| Feature | Current State | Recommended UI |
|---------|---------------|----------------|
| LLM model name | Hidden | Add input (advanced section) |
| Loop interval | Hidden | Add input (advanced section) |
| Synthetic skew settings | Hidden | Add in Calibration (advanced) |
| Position reconcile tolerance | Hidden | Add input (advanced section) |

---

## Next Steps

1. **Review this file** and decide which features should be exposed first based on operational needs.
2. **Create follow-up prompts** for implementing UI controls for the selected high-priority features.
3. **Re-run this audit** periodically as new features are added to the codebase.
4. **Consider runtime persistence** - some of these controls would need to save to env vars or a config store to survive restarts.

---

*Generated: December 2025*
*Last audit files: `src/config.py`, `src/risk_engine.py`, `src/web_app.py`, `src/healthcheck.py`, `src/agent_brain_llm.py`*
