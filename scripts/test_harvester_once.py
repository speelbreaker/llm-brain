#!/usr/bin/env python3
import sys
import os
import logging

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.data_harvester import fetch_option_chain, process_and_save, INVERSE_ASSETS

# Configure logging to stdout
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def test_run():
    asset = "BTC"
    print(f"Testing harvester for {asset}...")
    try:
        data = fetch_option_chain(asset)
        print(f"Fetched {len(data)} records for {asset}")
        
        if not data:
            print("No data returned! Check API availability.")
            return

        # Check first record structure
        print("First record sample:", data[0])
        
        # Test processing
        process_and_save(asset, data)
        print("Processing and save complete.")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_run()
