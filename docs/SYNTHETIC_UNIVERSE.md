# Synthetic Universe - Hybrid Mode System

This document describes the hybrid synthetic mode system for backtesting, which allows users to choose between different levels of realism when generating option data for simulations.

## Overview

The backtest system supports three modes for generating option data:

1. **Pure Synthetic (RV-based)** - Fully synthetic pricing using realized volatility
2. **Live IV, Synthetic Grid** - Uses live ATM IV from Deribit but generates synthetic strike grid
3. **Live Chain + Live IV** - Uses actual Deribit option chains with real mark IV

## Mode Configurations

### SigmaMode

Controls how volatility (sigma) is determined for option pricing:

| Mode | Description |
|------|-------------|
| `rv_x_multiplier` | Uses realized volatility × IV multiplier × skew adjustment |
| `atm_iv_x_multiplier` | Uses live ATM IV from Deribit × multiplier × skew adjustment |
| `mark_iv_x_multiplier` | Uses each option's actual mark IV from Deribit |

### ChainMode

Controls how the option chain (strikes/expiries) is constructed:

| Mode | Description |
|------|-------------|
| `synthetic_grid` | Generates a synthetic grid of strikes and expiries |
| `live_chain` | Fetches actual option chain from Deribit for the given timestamp |

## Preset Combinations

### Pure Synthetic (RV-based)
- **sigma_mode**: `rv_x_multiplier`
- **chain_mode**: `synthetic_grid`
- **Use case**: Fast backtests, historical periods without live data, research on volatility assumptions
- **Description**: Uses realized volatility with multiplier to price synthetic options on a generated strike grid.

### Live IV, Synthetic Grid
- **sigma_mode**: `atm_iv_x_multiplier`
- **chain_mode**: `synthetic_grid`
- **Use case**: Incorporate market's forward-looking IV expectations while maintaining consistent strike structure
- **Description**: Uses live ATM IV from Deribit with multiplier to price synthetic options on a generated strike grid.

### Live Chain + Live IV
- **sigma_mode**: `mark_iv_x_multiplier`
- **chain_mode**: `live_chain`
- **Use case**: Most realistic backtesting, validation against actual market conditions
- **Description**: Uses actual Deribit option chains with live mark IV for each strike. Most realistic backtesting.

## Implementation Details

### Key Files

- `src/backtest/types.py` - Contains `SigmaMode`, `ChainMode` type definitions and `CallSimulationConfig` with mode fields
- `src/backtest/pricing.py` - Implements `get_sigma_for_option()` which selects volatility based on sigma_mode
- `src/backtest/state_builder.py` - Implements chain construction logic based on chain_mode
- `src/backtest/manager.py` - Passes mode parameters through the backtest pipeline
- `src/web_app.py` - API endpoints and UI for selecting synthetic modes

### Sigma Selection Logic

```python
def get_sigma_for_option(config, option, chain, rv, iv_multiplier):
    if config.sigma_mode == "rv_x_multiplier":
        return rv * iv_multiplier * get_skew_adjustment(option.delta)
    elif config.sigma_mode == "atm_iv_x_multiplier":
        atm_iv = get_atm_iv_from_chain(chain)
        return atm_iv * iv_multiplier * get_skew_adjustment(option.delta)
    elif config.sigma_mode == "mark_iv_x_multiplier":
        return option.mark_iv  # Direct from Deribit
```

### Chain Construction Logic

```python
def build_candidates(config, timestamp, data_source):
    if config.chain_mode == "synthetic_grid":
        return generate_synthetic_grid(config, timestamp)
    elif config.chain_mode == "live_chain":
        return fetch_live_chain_from_deribit(config, timestamp)
```

## UI Integration

The Backtesting Lab includes a "Synthetic Mode" dropdown with the three presets. Selecting a preset automatically sets the appropriate `sigma_mode` and `chain_mode` values.

## Trade-offs

| Aspect | Pure Synthetic | Live IV Synthetic | Live Chain |
|--------|---------------|-------------------|------------|
| Speed | Fastest | Medium | Slowest |
| Data Requirements | None | ATM IV history | Full chain history |
| Realism | Low | Medium | High |
| Historical Coverage | Any period | Periods with IV data | Periods with chain snapshots |
| Strike Consistency | Perfect | Perfect | Varies by market |
