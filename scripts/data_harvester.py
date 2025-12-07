#!/usr/bin/env python3
"""
Deribit Data Harvester

A standalone script that continuously collects real Deribit options data
for all assets with active options markets and saves them as Parquet files
for backtesting and analysis.

Supports:
- Inverse options: BTC, ETH (settled in the underlying asset)
- Linear USDC options: SOL, XRP, AVAX, TRX, PAXG, etc. (settled in USDC)

This script runs independently from the trading bot / FastAPI server.

Usage:
    python -m scripts.data_harvester
"""

import datetime
import logging
import os
import sys
import time
from typing import Any

import pandas as pd
import requests

INVERSE_ASSETS = ["BTC", "ETH"]

LINEAR_CURRENCY = "USDC"

INTERVAL_MINUTES = int(os.getenv("HARVESTER_INTERVAL_MINUTES", "15"))

DERIBIT_BASE_URL = os.getenv("DERIBIT_BASE_URL", "https://www.deribit.com")

DATA_ROOT = os.getenv("HARVESTER_DATA_ROOT", "data/live_deribit")

LOG_DIR = "logs"

os.makedirs(DATA_ROOT, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger("data_harvester")
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler(os.path.join(LOG_DIR, "data_harvester.log"))
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
))

stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(logging.Formatter(
    "%(asctime)s - %(levelname)s - %(message)s"
))

logger.addHandler(file_handler)
logger.addHandler(stream_handler)

DESIRED_COLUMNS = [
    "harvest_time",
    "instrument_name",
    "expiration_timestamp",
    "strike",
    "option_type",
    "mark_price",
    "underlying_price",
    "underlying_index",
    "best_bid_price",
    "best_ask_price",
    "open_interest",
    "volume",
    "bid_iv",
    "ask_iv",
    "mark_iv",
    "greek_delta",
    "greek_gamma",
    "greek_theta",
    "greek_vega",
    "greek_rho",
]


def fetch_option_chain(currency: str) -> list[dict[str, Any]]:
    """
    Fetch option book summary data from Deribit for a given currency.
    
    Args:
        currency: The currency symbol (e.g., "BTC", "ETH")
        
    Returns:
        List of option data dictionaries, or empty list on error.
    """
    url = f"{DERIBIT_BASE_URL}/api/v2/public/get_book_summary_by_currency"
    params = {
        "currency": currency,
        "kind": "option",
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if "result" in data:
            return data["result"]
        else:
            logger.warning(f"No 'result' key in response for {currency}: {data.get('error', 'unknown error')}")
            return []
            
    except requests.exceptions.Timeout:
        logger.error(f"Timeout fetching {currency} options data")
        return []
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error fetching {currency}: {e}")
        return []
    except ValueError as e:
        logger.error(f"JSON decode error for {currency}: {e}")
        return []


def process_and_save(currency: str, raw_data: list[dict[str, Any]]) -> None:
    """
    Process raw option data and save to Parquet file.
    
    Args:
        currency: The currency symbol
        raw_data: List of option data dictionaries from Deribit API
    """
    if not raw_data:
        logger.info(f"No data to save for {currency}")
        return
    
    df = pd.DataFrame(raw_data)
    
    now = datetime.datetime.now(datetime.timezone.utc)
    df["harvest_time"] = now
    
    if "greeks" in df.columns:
        greeks_df = df["greeks"].apply(
            lambda x: pd.Series(x) if isinstance(x, dict) else pd.Series()
        )
        greeks_df = greeks_df.rename(columns=lambda c: f"greek_{c}")
        df = pd.concat([df.drop(columns=["greeks"]), greeks_df], axis=1)
    
    existing_cols = [c for c in DESIRED_COLUMNS if c in df.columns]
    df = df[existing_cols]
    
    year = now.strftime("%Y")
    month = now.strftime("%m")
    day = now.strftime("%d")
    time_str = now.strftime("%H%M")
    date_str = now.strftime("%Y-%m-%d")
    
    dir_path = os.path.join(DATA_ROOT, currency, year, month, day)
    os.makedirs(dir_path, exist_ok=True)
    
    filename = f"{currency}_{date_str}_{time_str}.parquet"
    filepath = os.path.join(dir_path, filename)
    
    df.to_parquet(filepath, engine="pyarrow", index=False)
    
    logger.info(f"Saved {len(df)} rows for {currency} to {filepath}")


def process_linear_options(raw_data: list[dict[str, Any]]) -> None:
    """
    Process USDC linear options and save grouped by underlying asset.
    
    Args:
        raw_data: List of option data dictionaries from USDC query
    """
    if not raw_data:
        logger.info("No linear USDC options data to save")
        return
    
    grouped: dict[str, list[dict[str, Any]]] = {}
    for option in raw_data:
        instrument_name = option.get("instrument_name", "")
        if "-" in instrument_name:
            underlying = instrument_name.split("-")[0]
            if underlying not in grouped:
                grouped[underlying] = []
            grouped[underlying].append(option)
    
    for underlying, options in grouped.items():
        logger.info(f"Processing {len(options)} options for {underlying}")
        process_and_save(underlying, options)


def run_harvester() -> None:
    """
    Main harvester loop. Runs indefinitely, fetching and saving option data
    for all configured assets at the specified interval.
    """
    logger.info("=" * 60)
    logger.info("Starting Deribit Data Harvester")
    logger.info(f"Inverse assets: {', '.join(INVERSE_ASSETS)}")
    logger.info(f"Linear currency: {LINEAR_CURRENCY}")
    logger.info(f"Polling interval: {INTERVAL_MINUTES} minutes")
    logger.info(f"Data directory: {DATA_ROOT}")
    logger.info(f"Deribit URL: {DERIBIT_BASE_URL}")
    logger.info("=" * 60)
    
    iteration = 0
    
    while True:
        iteration += 1
        loop_start = time.time()
        
        logger.info(f"--- Harvest iteration {iteration} ---")
        
        for asset in INVERSE_ASSETS:
            try:
                logger.info(f"Fetching inverse {asset}...")
                data = fetch_option_chain(asset)
                process_and_save(asset, data)
            except Exception as e:
                logger.exception(f"Unexpected error processing {asset}: {e}")
            
            time.sleep(0.5)
        
        try:
            logger.info(f"Fetching linear {LINEAR_CURRENCY} options...")
            linear_data = fetch_option_chain(LINEAR_CURRENCY)
            process_linear_options(linear_data)
        except Exception as e:
            logger.exception(f"Unexpected error processing linear options: {e}")
        
        elapsed = time.time() - loop_start
        sleep_sec = INTERVAL_MINUTES * 60 - elapsed
        
        if sleep_sec > 0:
            sleep_min = sleep_sec / 60
            logger.info(f"Harvest complete. Sleeping for {sleep_min:.1f} minutes...")
            time.sleep(sleep_sec)
        else:
            logger.warning(
                f"Harvesting took {elapsed:.1f}s, exceeding interval of "
                f"{INTERVAL_MINUTES * 60}s. Continuing immediately."
            )


if __name__ == "__main__":
    try:
        run_harvester()
    except KeyboardInterrupt:
        logger.info("Data Harvester stopped by user (KeyboardInterrupt)")
        sys.exit(0)
