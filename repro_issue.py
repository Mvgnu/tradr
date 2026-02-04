import yfinance as yf
import pandas as pd
import asyncio

async def test_download():
    symbol = "INVALID_SYMBOL_XYZ"
    print(f"Downloading data for {symbol}...")
    df = await asyncio.to_thread(lambda: yf.download(
        tickers=symbol,
        period="300d",
        interval="1d",
        progress=False,
        auto_adjust=False
    ))
    
    print(f"\nDataFrame Shape: {df.shape}")
    print(f"Columns: {df.columns}")
    if isinstance(df.columns, pd.MultiIndex):
        print("Columns are MultiIndex")
        flat_cols = ['_'.join([str(i) for i in col]).strip('_') for col in df.columns.values]
        print(f"Flattened columns: {flat_cols}")
    else:
        print("Columns are NOT MultiIndex")
        
    if not df.empty:
        print("\nFirst few rows:")
        print(df.head())
    else:
        print("\nDataFrame is empty!")

if __name__ == "__main__":
    asyncio.run(test_download())
