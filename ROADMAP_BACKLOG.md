# Options Trading Agent – Backlog

Ideas, improvements, and future work items not yet prioritized on the main roadmap.

---

## Data Infrastructure

### Data Harvesting & Collection
- [ ] **Long-term data retention** – Archive harvested parquet files to cloud storage (S3/GCS)
- [ ] **Data compression** – Implement parquet file compression for storage efficiency
- [ ] **Multi-region harvesting** – Run harvesters in different regions for redundancy
- [ ] **Mainnet vs testnet toggle** – Easy switch between mainnet and testnet data collection
- [ ] **Historical data backfill** – Script to fetch historical data from Tardis or similar providers

### Data Quality
- [ ] **Data validation pipeline** – Automated checks for missing fields, outliers, stale prices
- [ ] **Gap detection** – Alert when harvesting has gaps (missed intervals)
- [ ] **Data freshness dashboard** – Show last harvest time per underlying in UI

---

## Backtesting Improvements

### Performance & Accuracy
- [ ] **Slippage modeling** – Simulate realistic fill prices vs mark prices
- [ ] **Fee models** – Configurable exchange fees, funding rates
- [ ] **Latency simulation** – Model order execution delays
- [ ] **Multi-asset correlation** – Consider BTC/ETH correlation in risk calculations

### Comparison & Analysis
- [ ] **Enrich metrics** – Populate USD drawdown, avg trade fields in backtest results
- [ ] **Timestamped JSON reports** – Include date range in health check filenames to avoid overwrites
- [ ] **Cross-strategy comparison** – Compare multiple strategies in a single report
- [ ] **Regime-aware analysis** – Break down performance by market regime (bull/bear/sideways)

### Testing Infrastructure
- [ ] **Unit tests for diff/compare modules** – Cover edge cases (mismatched exit styles, missing data)
- [ ] **Integration tests** – End-to-end backtest pipeline validation
- [ ] **Benchmark suite** – Track backtest execution time to prevent regressions

---

## Web Dashboard Enhancements

### Visualization
- [ ] **Equity curve comparison** – Overlay multiple backtests on single chart
- [ ] **Options Greeks over time** – Chart portfolio delta, theta evolution
- [ ] **Trade annotation** – Click on equity curve to see trade details
- [ ] **Calendar heatmap** – Daily PnL visualization

### User Experience
- [ ] **Backtest presets** – Save/load common backtest configurations
- [ ] **Export to CSV** – Download backtest chains/trades as spreadsheet
- [ ] **Dark mode** – User-selectable theme
- [ ] **Mobile responsive** – Better layout for tablet/phone viewing

### Monitoring
- [ ] **Alert system** – Email/webhook notifications for agent events
- [ ] **System health page** – Show DB connection status, harvester uptime, API latency
- [ ] **Audit log** – Track all agent actions with timestamps

---

## Trading Strategy Ideas

### New Strategies
- [ ] **Cash-secured puts** – Sell puts instead of covered calls
- [ ] **Iron condors** – Multi-leg options spreads
- [ ] **Calendar spreads** – Different expiry combinations
- [ ] **Straddles/strangles** – Volatility plays

### Strategy Improvements
- [ ] **Dynamic delta targeting** – Adjust target delta based on volatility regime
- [ ] **Premium threshold** – Minimum premium to open a position
- [ ] **Expiry clustering avoidance** – Spread risk across expiry dates
- [ ] **Correlation-aware sizing** – Reduce size when BTC/ETH move together

---

## LLM & AI Features

### Fine-tuning Pipeline
- [ ] **Automated training data refresh** – Regenerate training data weekly
- [ ] **Model evaluation framework** – Compare fine-tuned vs base model performance
- [ ] **Prompt versioning** – Track and compare different prompt strategies
- [ ] **Cost tracking** – Monitor OpenAI API spend per feature

### Research Mode
- [ ] **Automated experiment runner** – LLM proposes, system tests automatically
- [ ] **A/B testing framework** – Compare live strategies side-by-side
- [ ] **Performance attribution** – Analyze what factors drive returns

---

## Risk Management

### Enhanced Controls
- [ ] **Portfolio-level Greeks limits** – Cap total delta/gamma across all positions
- [ ] **Correlation stress tests** – Simulate correlated moves across underlyings
- [ ] **Black swan protection** – Automatic position reduction during extreme volatility
- [ ] **Position aging alerts** – Flag positions approaching expiry

### Monitoring
- [ ] **Real-time margin utilization** – Show percentage of available margin used
- [ ] **Liquidation price alerts** – Warn when spot approaches danger zones
- [ ] **Historical risk metrics** – Track portfolio VaR over time

---

## Infrastructure & DevOps

### Reliability
- [ ] **Health check endpoint** – `/health` for load balancer integration
- [ ] **Graceful shutdown** – Save state before workflow stops
- [ ] **Auto-recovery** – Restart agent on failure with exponential backoff
- [ ] **Database backups** – Automated PostgreSQL snapshots

### Observability
- [ ] **Structured logging to external service** – Send logs to Datadog/Grafana
- [ ] **Metrics collection** – Prometheus-compatible metrics endpoint
- [ ] **Distributed tracing** – Track request flow through system

### Security
- [ ] **API rate limiting** – Protect endpoints from abuse
- [ ] **Secret rotation** – Automated API key refresh
- [ ] **Audit logging** – Track all sensitive operations

---

## Documentation

- [ ] **API documentation** – OpenAPI/Swagger spec for all endpoints
- [ ] **Training mode guide** – Step-by-step for setting up training runs
- [ ] **Deployment guide** – Instructions for mainnet deployment
- [ ] **Troubleshooting guide** – Common issues and solutions
- [ ] **Architecture decision records** – Document key design choices

---

## Technical Debt

### Code Quality
- [ ] **Clean up LSP diagnostics** – Resolve remaining type hints and unused imports
- [ ] **Consolidate data source handling** – Unify SYNTHETIC/LIVE_DERIBIT/REAL_SCRAPER loading
- [ ] **Remove deprecated files** – Clean up old/unused scripts
- [ ] **Consistent error handling** – Standardize exception patterns across modules

### Performance
- [ ] **Optimize database queries** – Add indexes for common query patterns
- [ ] **Cache frequently accessed data** – Redis or in-memory caching for spot prices
- [ ] **Lazy loading** – Defer expensive computations until needed

---

## Integration Ideas

- [ ] **Telegram bot** – Receive trade notifications via Telegram
- [ ] **Discord integration** – Post daily performance summaries
- [ ] **Google Sheets export** – Sync backtest results to spreadsheet
- [ ] **TradingView signals** – Publish signals as TradingView alerts

---

## Notes

This backlog is intentionally broad. Items here are not prioritized and may never be implemented. The purpose is to capture ideas for future consideration.

To promote an item to the main roadmap:
1. Evaluate effort vs. value
2. Check dependencies on other work
3. Add to appropriate phase in ROADMAP.md

---

*Last updated: December 2024*
