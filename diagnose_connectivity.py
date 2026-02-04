import sys
import os
import yfinance as yf
import requests
import asyncio

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tradr.core.config import setup_clients

async def diagnose():
    print("--- Starting Connectivity Diagnosis ---")
    
    # 1. Test yfinance
    print("\n[1/3] Testing Yahoo Finance (yfinance)...")
    try:
        loop = asyncio.get_event_loop()
        df = await loop.run_in_executor(
            None,
            lambda: yf.download(tickers="SPY", period="1d", interval="1d", progress=False)
        )
        if not df.empty:
            print("✅ yfinance: SUCCESS - Data downloaded for SPY.")
        else:
            print("❌ yfinance: FAILED - Download returned empty dataframe.")
    except Exception as e:
        print(f"❌ yfinance: FAILED - Exception: {e}")

    # 2. Test Finnhub
    print("\n[2/3] Testing Finnhub...")
    clients = setup_clients()
    finnhub_key = clients.get("finnhub_key")
    if finnhub_key:
        try:
            response = requests.get(
                "https://finnhub.io/api/v1/news",
                params={"category": "general", "token": finnhub_key},
                timeout=10
            )
            if response.status_code == 200 and response.json():
                print("✅ Finnhub: SUCCESS - API key is valid and news received.")
            else:
                print(f"❌ Finnhub: FAILED - Status Code: {response.status_code}, Response: {response.text}")
        except Exception as e:
            print(f"❌ Finnhub: FAILED - Exception: {e}")
    else:
        print("⚪ Finnhub: SKIPPED - FINNHUB_KEY not found in environment.")

    # 3. Test Alpaca
    print("\n[3/3] Testing Alpaca Markets...")
    alpaca_client = clients.get("alpaca")
    if alpaca_client:
        try:
            loop = asyncio.get_event_loop()
            account = await loop.run_in_executor(None, alpaca_client.get_account)
            if account.id:
                print(f"✅ Alpaca: SUCCESS - Connected to account {account.id} (Status: {account.status})")
            else:
                print("❌ Alpaca: FAILED - Could not retrieve account ID.")
        except Exception as e:
            print(f"❌ Alpaca: FAILED - Exception: {e}")
    else:
        print("⚪ Alpaca: SKIPPED - Alpaca client not initialized.")
        
    print("\n--- Diagnosis Complete ---")

if __name__ == "__main__":
    asyncio.run(diagnose())