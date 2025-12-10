# Options Trading Agent - Deribit Testnet

## Overview
This project is a modular Python framework for automated BTC/ETH covered call trading on the Deribit testnet. It serves as a research and experimentation system for testing trading strategies, generating training data, and analyzing performance through a web dashboard and backtesting suite. The system supports both rule-based and LLM-powered decision-making, with a focus on exploration-based learning and ambitions for mainnet deployment.

## User Preferences
- Python 3.11
- Type hints everywhere
- Pydantic for configs and models
- httpx for HTTP
- Clarity over cleverness

## System Architecture
The agent features a clear separation of concerns, with modules for configuration, data modeling, API interaction, market context generation, risk management, policy decisions (rule-based and LLM), execution, and logging. It supports "research" and "production" modes, with a FastAPI web application providing a real-time dashboard for monitoring, interaction, and backtesting.

### UI/UX Decisions
The web dashboard provides a user-friendly interface with sections for "Live Agent" status, "Backtesting Lab" with interactive charts, "Backtest Runs" management, "Calibration" for price comparison, "System Health" with runtime controls, and a "Chat" interface for natural language interaction. It also includes "Bots" tab for expert trading bot analysis and an "AI Steward" panel for project insights.

### Technical Implementations
- **Configuration**: Pydantic settings manage application configuration.
- **API Wrapper**: `deribit_client.py` provides an `httpx`-based wrapper for the Deribit testnet API.
- **State Management**: `state_builder.py` aggregates market data; `status_store.py` manages thread-safe status updates. `state_core.py` provides unified state-building logic for live and backtest agents.
- **Order Execution**: `execution.py` handles order translation and dry-run simulations.
- **Structured Logging**: Uses JSONL for logging decisions and actions.
- **Position Persistence**: Bot-managed positions are saved to `data/positions.json` and restored on restart.
- **Agent Healthcheck Module**: Self-contained healthcheck system that exercises the critical pipeline (config → Deribit → state builder).

### Feature Specifications
- **Decision Policies**: Supports rule-based strategies with scoring and epsilon-greedy exploration, and an LLM-powered decision mode validated by a risk engine. Decision modes include `rule_only`, `llm_only`, and `hybrid_shadow`.
- **LLM Validation**: `validate_llm_decision()` function provides comprehensive validation including symbol verification, action type checking, position verification, and size clamping.
- **Market Context**: Integrates `MarketContext` for regime detection, returns, and realized volatility.
- **Risk Management**: A `risk_engine` performs pre-trade validation, checking margin, delta, exposure limits, liquidity guards, and hard safety rails (global kill switch, daily drawdown guard).
- **Backtesting Framework**: Includes `CoveredCallSimulator` for historical analysis, supporting various exit styles and training data generation. Features a synthetic pricing mode and provides TradingView-style metrics and equity curve visualization.
- **Selector Analysis**: Includes "Selector Frequency Scan" for analyzing rule pass rates and "Selector Heatmap" for 2D parameter sweep visualization.
- **Environment Heatmap**: Provides a selector-free 2D occupancy heatmap to show market time spent across metric pairs. A CLI script (`scripts/run_greg_environment_heatmaps.py`) and UI panel automate environment sweet spot analysis.
- **Training Mode**: Allows multi-profile data collection for ML/RL, with aggressive policies and configurable per-expiry limits. Exports training data in chain-level and candidate-level formats.
- **LLM Fine-Tuning Data**: Scripts transform candidate CSVs into chat-style JSONL corpora for LLM fine-tuning.
- **Runtime Controls**: System Health tab offers interactive controls for adjusting safety and operational settings (e.g., Global Kill Switch, Daily Drawdown Limit, Decision Mode, Dry Run Mode, Liquidity Guards, Position Reconciliation Config).
- **LLM & Strategy Tuning**: System Health tab includes a panel for adjusting LLM and strategy configuration at runtime (e.g., LLM Enabled toggle, Explore Probability, Training Profile, Strategy Thresholds, Risk Limits).
- **Persistent Backtest Runs**: Stores backtest results in PostgreSQL using SQLAlchemy ORM.
- **Comparison & Health Check Scripts**: Tools for comparing synthetic vs. live backtests and performing strategy health checks.
- **AI Steward (Project Brain)**: A project planning and QA helper (`src/system_steward.py`) that summarizes project state and suggests next tasks using an LLM, exposed via API endpoints and a UI panel.
- **State-Aware Chat Assistant**: A multi-turn assistant that understands current trading state, answers questions, and provides project information.
- **Position Reconciliation**: Compares local position tracker against exchange positions, with configurable actions.
- **Calibration vs Deribit**: Compares synthetic Black-Scholes prices against live Deribit marks.
- **Synthetic Universe v2 (Greg-sensor cluster regimes)**:
  - `src/synthetic/regimes.py` with `RegimeParams`, KMeans clustering, and AR(1) IV dynamics
  - Uses Greg-sensor clusters (VRP, ADX, chop, IV rank, term slope, skew) to infer volatility regimes
  - AR(1) IV dynamics with VRP target + skew template for realistic IV evolution
  - `scripts/build_greg_regimes_from_harvester.py` to calibrate regimes from real Deribit data
  - Regime model saved to `data/greg_regimes.json` (per underlying, with transition matrix)
  - Backtest pricing module extended with `RegimeState` and regime-aware IV functions
- **Extended Calibration System (v2)**:
  - `src/calibration_config.py`: Pydantic models for CalibrationConfig, HarvestConfig, BandConfig, CalibrationFilters
  - `src/calibration_extended.py`: Enhanced run_calibration_extended() with liquidity filtering, multi-DTE bands, bucket metrics, skew fitting, recommended vol_surface generation
  - **Option Types**: `CalibrationConfig.option_types` field controls which option types to calibrate: `["C"]` for calls only (default), `["P"]` for puts only, `["C", "P"]` for both
  - **Default Term Bands**: When `bands` is not provided, uses default weekly (3-10d), monthly (20-40d), quarterly (60-100d) bands for reporting
  - **Per-Type Metrics**: `by_option_type` field in results provides separate MAE/bias/bands for calls vs puts
  - **UI Calibration Coverage**: Shows option types used, per-type metrics table, and term structure buckets in the Calibration tab
  - `src/synthetic/vol_surface.py`: VolSurfaceConfig with DTE-band-specific IV multipliers and skew templates
  - `scripts/update_vol_surface_from_calibration.py`: CLI to run calibration and generate vol_surface config
  - `scripts/realism_check.py`: Realism checker comparing synthetic vs harvested distributions
  - Supports `source: "live" | "harvested"` for API or Parquet data
  - Global metrics: mae_vol_points, vega_weighted_mae_pct, residuals_summary
  - Skew fitting with anchor_ratios computation and misfit detection
  - RegimeParams extended with `calibrated_iv_multiplier` and `calibrated_skew_scale` fields
  - RegimeModel extended with `predict_cluster()` and `get_regime_occupancy()` methods
  - realism_check.py uses RegimeModel for regime assignment and transition matrix comparison
- **Auto IV Calibration Pipeline**: 
  - **calibration_history table**: Persists time series of auto-calculated IV multipliers keyed by underlying and DTE range.
  - **scripts/auto_calibrate_iv.py**: CLI script that loads harvester data, fits an IV multiplier minimizing MAE, and stores results in the database.
  - **calibration_store.py**: Runtime in-memory override store for IV multipliers (resets on restart).
  - **API endpoints**: `GET /api/calibration/history` returns recent calibrations; `POST /api/calibration/use_latest` applies the latest multiplier as a runtime override.
  - **Calibration tab UI**: Includes "Use Latest Recommended" button and "Calibration History" table to view and apply historical calibrations.
- **Calibration Update Policy System**:
  - **src/calibration_update_policy.py**: Policy layer with smoothing (EWMA), thresholds (min_delta, min_sample_size, min_vega_sum), and file-based history storage.
  - **History Storage**: JSON files saved to `data/calibration_runs/<timestamp>_<underlying>_<source>.json` with full run details, smoothed values, and apply decisions.
  - **CalibrationUpdatePolicy**: Configurable thresholds (min_delta_global=0.03, min_sample_size=50, smoothing_window_days=14).
  - **Decision Logic**: `should_apply_update()` checks sample size, vega sum, and delta thresholds before applying.
  - **Smoothing**: EWMA over configurable window to prevent overreacting to noisy days.
  - **CLI Integration**: `scripts/update_vol_surface_from_calibration.py` now uses policy layer with `--force` and `--no-policy` flags.
  - **API Endpoints**: 
    - `GET /api/calibration/policy`: Returns current policy thresholds and explanation.
    - `GET /api/calibration/current_multipliers`: Returns currently applied IV multipliers.
    - `GET /api/calibration/runs`: Returns recent calibration runs with apply status.
    - `POST /api/calibration/force_apply`: Force-apply calibration bypassing thresholds.
    - `POST /api/calibration/run_with_policy`: Run calibration respecting policy thresholds.
  - **UI Panel**: "IV Calibration Update Policy" section in Calibration tab showing current multipliers, latest run status, policy explanation, and "Force-Apply" button.
- **Harvester Data Quality & Reproducibility**:
  - `src/harvester/health.py`: Schema validation and quality assessment for harvested Parquet snapshots
  - `REQUIRED_COLUMNS`: Canonical schema definition with expected column types
  - `validate_snapshot_schema()`: Returns list of schema issues (missing columns, wrong types)
  - `assess_snapshot_quality()`: Returns SnapshotQualityReport with row counts and core-field completeness
  - `aggregate_quality_reports()`: Aggregates multiple reports into DataQualitySummary with overall status
  - **DataQualityStatus**: OK / DEGRADED / FAILED status based on schema and quality thresholds
  - **Historical Calibration Integration**: `run_historical_calibration_from_harvest()` now includes:
    - `data_quality` block with snapshot counts, schema issues, low-quality counts, and completeness
    - `reproducibility` metadata with harvest_config, calibration_config_hash, greg_regimes_version
  - **Realism Checker Integration**: `realism_check.py` now prints Data Health summary and adjusts score for quality issues
  - **UI Panel**: "Data Health & Reproducibility" section in Calibration tab showing:
    - Status badge (OK/DEGRADED/FAILED with color coding)
    - Metrics: snapshots, schema issues, low-quality count, core completeness
    - Issues list when problems detected
    - Reproducibility info: run time, underlying, harvest period, config hash, regime model version
    - "View Raw Metadata" button for detailed JSON inspection
- **Bots System**: Provides a comprehensive view of expert trading bots, market sensors, and strategy evaluations, including debug mode for sensor computations.
    - **Greg Mandolini VRP Harvester (GregBot) v6.0 "Diamond-Grade"**: A quantitative VRP strategy selector based on 11 volatility sensors and a decision tree. Currently advisory (read-only) with 8 evaluated strategies per underlying. Sensor mapping and calibration variables are defined.
- **Strategy Layer**: A pluggable architecture allowing multiple trading strategies to run concurrently (`src/strategies/`). Strategies are built from settings via `build_default_registry()` and decisions include `strategy_id` for attribution.

## External Dependencies
- **Deribit API**: Used for real-time market data (testnet) and historical data (mainnet public API for backtesting and data harvesting).
- **OpenAI**: Integrated for LLM-powered decision-making and generating insights.
- **PostgreSQL**: Used for persistent storage of backtest runs.