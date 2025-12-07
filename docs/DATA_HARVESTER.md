# Deribit Data Harvester

A standalone script that continuously collects real Deribit options data for multiple assets and saves them as Parquet files for backtesting and analysis.

## Overview

The Data Harvester runs as a **separate process** from the trading bot / FastAPI server. It polls the Deribit mainnet public API at regular intervals and stores option chain snapshots in a structured directory tree.

## Features

- Fetches option book summary data for multiple assets: BTC, ETH, SOL, XRP, DOGE, MATIC
- Saves snapshots as Parquet files with timestamps
- Includes Greeks (delta, gamma, theta, vega) when available
- Robust error handling - continues running on transient failures
- Configurable polling interval (default: 15 minutes)

## Installation

Ensure dependencies are installed:

```bash
pip install -r requirements.txt
```

Required packages:
- pandas
- pyarrow
- requests

## Usage

Run the harvester as a standalone process:

```bash
python -m scripts.data_harvester
```

The script will:
1. Start fetching data for all configured assets
2. Save Parquet snapshots to `data/live_deribit/`
3. Sleep until the next interval
4. Repeat indefinitely

## Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `DERIBIT_BASE_URL` | `https://www.deribit.com` | Deribit API base URL |
| `HARVESTER_INTERVAL_MINUTES` | `15` | Polling interval in minutes |
| `HARVESTER_DATA_ROOT` | `data/live_deribit` | Root directory for saved data |

## Output Structure

Data is saved in a hierarchical directory structure:

```
data/live_deribit/
├── BTC/
│   └── 2025/
│       └── 12/
│           └── 07/
│               ├── BTC_2025-12-07_1400.parquet
│               ├── BTC_2025-12-07_1415.parquet
│               └── ...
├── ETH/
│   └── 2025/
│       └── ...
└── ...
```

## Data Schema

Each Parquet file contains:

| Column | Description |
|--------|-------------|
| `harvest_time` | UTC timestamp when data was fetched |
| `instrument_name` | Option instrument (e.g., `BTC-27DEC25-100000-C`) |
| `expiration_timestamp` | Expiry timestamp (ms) |
| `strike` | Strike price |
| `option_type` | `call` or `put` |
| `mark_price` | Mark price (in underlying) |
| `underlying_price` | Spot/index price |
| `best_bid_price` | Best bid |
| `best_ask_price` | Best ask |
| `open_interest` | Open interest |
| `volume` | 24h volume |
| `bid_iv` | Bid implied volatility |
| `ask_iv` | Ask implied volatility |
| `mark_iv` | Mark implied volatility |
| `greek_delta` | Option delta |
| `greek_gamma` | Option gamma |
| `greek_theta` | Option theta |
| `greek_vega` | Option vega |

## Logs

Logs are written to:
- `logs/data_harvester.log` (file)
- stdout (console)

## Running in Production

For long-running deployment, consider:

1. **Screen/tmux session:**
   ```bash
   screen -S harvester
   python -m scripts.data_harvester
   # Ctrl+A, D to detach
   ```

2. **Systemd service** (Linux):
   ```ini
   [Unit]
   Description=Deribit Data Harvester
   After=network.target

   [Service]
   ExecStart=/usr/bin/python3 -m scripts.data_harvester
   WorkingDirectory=/path/to/repo
   Restart=always
   User=youruser

   [Install]
   WantedBy=multi-user.target
   ```

3. **Separate Replit workflow** (if running on Replit)

## Notes

- This script uses Deribit's **public mainnet API** (no authentication required)
- The trading bot and harvester run independently - if one crashes, the other continues
- Data can be used for backtesting, ML training, or "exam mode" validation
