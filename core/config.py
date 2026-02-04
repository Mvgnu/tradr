import os
import sys
from dotenv import load_dotenv
import google.generativeai as genai
from sec_api import QueryApi
from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient

load_dotenv()

CONFIG = {
    "llm_model": "gemini-2.5-flash",
    "main_loop_sleep_seconds": 300,
    "max_position_size_pct": 15.0,
    "max_sector_exposure_pct": 30.0,
    "max_drawdown_pct": 20.0,
    "target_positions": 8,
    "min_sharpe_ratio": 0.5,
    "ignore_market_hours": False,
    "max_trade_participation_rate": 0.025,  # Do not trade more than 2.5% of avg daily volume
}

CONFIG.update({
    "default_vix_level": 20.0,
    "vix_low_threshold": 15.0,
    "vix_high_threshold": 25.0,
    "bull_breadth_threshold": 60.0,
    "bear_breadth_threshold": 40.0,
    # Volume analysis thresholds
    "volume_surge_ratio_20": 1.5,
    "volume_surge_ratio_50": 1.3,
    "volume_increasing_ratio_20": 1.2,
    "volume_increasing_ratio_50": 1.1,
    "volume_declining_ratio_20": 0.7,
    "volume_declining_ratio_50": 0.8,
    "volume_drying_up_ratio_20": 0.5,
    # Position sizing multipliers (from calculate_position_size)
    "volume_adjustment_high": 1.25,
    "volume_adjustment_medium": 1.1,
    "volume_adjustment_low": 0.7,
    # Risk management thresholds
    "max_risk_per_trade_pct": 2.0,
    "min_position_size_pct": 0.5,
    "max_position_size_pct": 15.0,
    # Technical analysis thresholds
    "rsi_oversold": 30,
    "rsi_overbought": 70,
    "trend_confirmation_days": 3,
    "stop_loss_atr_multiple": 1.5,
    "take_profit_atr_multiple": 2.5,
})

SECTOR_MAP = {
    "AAPL": "Technology", "MSFT": "Technology", "GOOGL": "Technology", "GOOG": "Technology",
    "AMZN": "Consumer Discretionary", "META": "Technology", "TSLA": "Consumer Discretionary", "NVDA": "Technology",
    "NFLX": "Communication Services", "CRM": "Technology", "ORCL": "Technology", "ADBE": "Technology",
    "PYPL": "Technology", "INTC": "Technology", "AMD": "Technology", "QCOM": "Technology",
    "UBER": "Technology", "SHOP": "Technology", "SQ": "Technology", "ROKU": "Technology",
    "HD": "Consumer Discretionary", "MCD": "Consumer Discretionary", "NKE": "Consumer Discretionary", "SBUX": "Consumer Discretionary",
    "LOW": "Consumer Discretionary", "TJX": "Consumer Discretionary", "F": "Consumer Discretionary",
    "GM": "Consumer Discretionary", "DIS": "Communication Services", "BKNG": "Consumer Discretionary",
    "JNJ": "Healthcare", "PFE": "Healthcare", "UNH": "Healthcare", "ABBV": "Healthcare",
    "TMO": "Healthcare", "DHR": "Healthcare", "BMY": "Healthcare", "LLY": "Healthcare",
    "MDT": "Healthcare", "AMGN": "Healthcare", "GILD": "Healthcare", "CVS": "Healthcare",
    "JPM": "Financials", "BAC": "Financials", "WFC": "Financials", "GS": "Financials",
    "MS": "Financials", "C": "Financials", "AXP": "Financials", "BLK": "Financials",
    "SCHW": "Financials", "USB": "Financials", "PNC": "Financials", "TFC": "Financials",
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy", "EOG": "Energy",
    "SLB": "Energy", "PSX": "Energy", "VLO": "Energy", "MPC": "Energy",
    "KMI": "Energy", "OKE": "Energy", "WMB": "Energy", "EPD": "Energy",
    "PG": "Consumer Staples", "KO": "Consumer Staples", "PEP": "Consumer Staples",
    "WMT": "Consumer Staples", "COST": "Consumer Staples", "CL": "Consumer Staples",
    "KHC": "Consumer Staples", "GIS": "Consumer Staples", "K": "Consumer Staples",
    "BA": "Industrials", "CAT": "Industrials", "UNP": "Industrials", "HON": "Industrials",
    "UPS": "Industrials", "LMT": "Industrials", "MMM": "Industrials", "GE": "Industrials",
    "RTX": "Industrials", "DE": "Industrials", "FDX": "Industrials", "NOC": "Industrials",
    "LIN": "Materials", "APD": "Materials", "ECL": "Materials", "FCX": "Materials",
    "NEM": "Materials", "DOW": "Materials", "DD": "Materials", "PPG": "Materials",
    "AMT": "Real Estate", "PLD": "Real Estate", "CCI": "Real Estate", "EQIX": "Real Estate",
    "PSA": "Real Estate", "WELL": "Real Estate", "DLR": "Real Estate", "O": "Real Estate",
    "NEE": "Utilities", "SO": "Utilities", "DUK": "Utilities", "AEP": "Utilities",
    "SRE": "Utilities", "D": "Utilities", "EXC": "Utilities", "XEL": "Utilities",
    "VZ": "Communication Services", "T": "Communication Services",
    "CHTR": "Communication Services", "CMCSA": "Communication Services", "TMUS": "Communication Services"
}

def get_api_keys():
    keys = {
        "ALPACA_KEY": os.getenv("ALPACA_KEY"),
        "ALPACA_SECRET": os.getenv("ALPACA_SECRET"),
        "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY"),
        "SEC_API_KEY": os.getenv("SEC_API_KEY"),
        "FINNHUB_KEY": os.getenv("FINNHUB_KEY"),
    }
    
    if not all([keys["ALPACA_KEY"], keys["ALPACA_SECRET"], keys["GOOGLE_API_KEY"]]):
        sys.exit("❌  Set ALPACA_KEY, ALPACA_SECRET, and GOOGLE_API_KEY in your env.")
    
    return keys


def setup_clients():
    keys = get_api_keys()
    
    alpaca_client = TradingClient(keys["ALPACA_KEY"], keys["ALPACA_SECRET"], paper=True)
    alpaca_data_client = StockHistoricalDataClient(keys["ALPACA_KEY"], keys["ALPACA_SECRET"])
    
    genai.configure(api_key=keys["GOOGLE_API_KEY"])
    
    queryapi = QueryApi(api_key=keys["SEC_API_KEY"]) if keys["SEC_API_KEY"] else None
    
    return {
        "alpaca": alpaca_client,
        "alpaca_data": alpaca_data_client,
        "queryapi": queryapi,
        "finnhub_key": keys["FINNHUB_KEY"]
    }
