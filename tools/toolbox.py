import asyncio
import logging
from typing import Dict, List, Any, Callable
import time
import requests
import os
import datetime as dt
import pandas as pd
import warnings
import yfinance as yf
from functools import wraps
import tiktoken
import json
from tradr.core.config import CONFIG, SECTOR_MAP
from tradr.core.position_sizing import calculate_position_size as core_calculate_position_size
from tradr.tools.cache import cache, _cache
import numpy as np
from collections import defaultdict

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    GetOptionContractsRequest,
    MarketOrderRequest,
    LimitOrderRequest,
    GetPortfolioHistoryRequest,
    TakeProfitRequest,
    StopLossRequest,
    StopOrderRequest,
    GetOrdersRequest,
)
from alpaca.data.requests import StockLatestQuoteRequest, StockLatestBarRequest
from alpaca.trading.enums import (
    OrderSide,
    TimeInForce,
    ContractType,
    AssetStatus,
    OrderClass,
    OrderType,
    QueryOrderStatus,
)
from tradr.core.performance import calculate_twrr
import random

# Suppress pandas_ta deprecation warnings
import yfinance.shared as yf_shared

# Direct imports - avoiding circular dependencies

# Direct imports - avoiding circular dependencies
try:
    from tradr.tools.watchers import WatcherManager
    from tradr.tools.scanners import MarketScanner
    from tradr.tools.calendar import EarningsCalendar
    from tradr.tools.utils import (
        validate_smart_order, 
        dedupe_blob, 
        rotate_tool_usage_log,
        strict_json_schema_validate,
        get_close_col
    )
except ImportError:
    # Fallback for testing
    WatcherManager = None
    MarketScanner = None
    EarningsCalendar = None
    validate_smart_order = lambda x: {"valid": True}
    dedupe_blob = lambda x: {"is_duplicate": False}
    rotate_tool_usage_log = lambda x: {"rotated": 0}
    strict_json_schema_validate = lambda x, y: {"valid": True}

class TokenBudgetError(Exception):
    pass

def with_budget(max_tokens: int = 50000):
    """Token budget decorator to prevent expensive API calls."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            result = await func(*args, **kwargs)
            
            if isinstance(result, str):
                text = result
            elif isinstance(result, dict):
                text = str(result)
            elif isinstance(result, list):
                text = str(result)
            else:
                text = str(result)
            
            try:
                encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
                token_count = len(encoding.encode(text))
                
                if token_count > max_tokens:
                    logging.warning(f"Token count {token_count} exceeds budget {max_tokens}")
                    import google.generativeai as genai
                    
                    model = genai.GenerativeModel('gemini-2.5-flash')
                    summary_prompt = f"Summarize this data concisely (max {max_tokens//4} tokens):\n\n{text[:10000]}"
                    response = model.generate_content(summary_prompt)
                    
                    if isinstance(result, dict):
                        return {"summary": response.text, "original_tokens": token_count}
                    elif isinstance(result, list):
                        return [{"summary": response.text, "original_tokens": token_count}]
                    else:
                        return response.text
                        
            except Exception as e:
                logging.warning(f"Token counting failed: {e}")
                
            return result
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            
            if isinstance(result, str):
                text = result
            elif isinstance(result, dict):
                text = str(result)
            elif isinstance(result, list):
                text = str(result)
            else:
                text = str(result)
            
            try:
                encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
                token_count = len(encoding.encode(text))
                
                if token_count > max_tokens:
                    logging.warning(f"Token count {token_count} exceeds budget {max_tokens}")
                    import google.generativeai as genai
                    
                    model = genai.GenerativeModel('gemini-2.5-flash')
                    summary_prompt = f"Summarize this data concisely (max {max_tokens//4} tokens):\n\n{text[:10000]}"
                    response = model.generate_content(summary_prompt)
                    
                    if isinstance(result, dict):
                        return {"summary": response.text, "original_tokens": token_count}
                    elif isinstance(result, list):
                        return [{"summary": response.text, "original_tokens": token_count}]
                    else:
                        return response.text
                        
            except Exception as e:
                logging.warning(f"Token counting failed: {e}")
                
            return result
        
        # Return the appropriate wrapper based on whether the function is async
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    return decorator

class AsyncToolbox:
    """Async-enabled toolbox for concurrent API calls."""
    
    def __init__(self, clients: Dict[str, Any], memory=None, executor=None):
        self.clients = clients
        self.memory = memory  # Store reference to memory system
        self.executor = executor
        self.rate_limiter = {'last_call': {}, 'min_interval': 0.5}
        self.tool_usage_log = []
        self.semaphores = {
            'yfinance': asyncio.Semaphore(5),
            'sec': asyncio.Semaphore(2),
            'finnhub': asyncio.Semaphore(3),
            'alpaca': asyncio.Semaphore(10)
        }
        
        # Initialize additional components
        self.watcher_manager = WatcherManager(self) if WatcherManager else None
        self.market_scanner = MarketScanner(clients) if MarketScanner else None
        self.earnings_calendar = EarningsCalendar(clients) if EarningsCalendar else None

    def _calculate_rsi(self, series: pd.Series, length: int = 14) -> pd.Series:
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).ewm(com=length - 1, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(com=length - 1, adjust=False).mean()
        # Handle division by zero
        rs = gain / loss
        rs[loss == 0] = np.inf # If no loss, RS is infinite
        rs[gain == 0] = 0      # If no gain, RS is zero
        # Handle cases where both gain and loss are zero (flat price)
        rs[(gain == 0) & (loss == 0)] = 0 # RSI is 0 in this case for flat
        return 100 - (100 / (1 + rs))

    def _calculate_macd(self, series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        exp1 = series.ewm(span=fast, adjust=False).mean()
        exp2 = series.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        histogram = macd - signal_line
        df_macd = pd.DataFrame({'MACD': macd, 'MACDs': signal_line, 'MACDh': histogram})
        return df_macd

    def _calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
        high_low = high - low
        high_close = (high - close.shift()).abs()
        low_close = (low - close.shift()).abs()
        tr = pd.DataFrame({'HL': high_low, 'HC': high_close, 'LC': low_close}).max(axis=1)
        return tr.ewm(com=length - 1, adjust=False).mean()

    def _calculate_obv(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        obv = pd.Series(0.0, index=close.index)
        # Use .loc to avoid SettingWithCopyWarning
        obv.loc[close > close.shift(1)] = volume.loc[close > close.shift(1)]
        obv.loc[close < close.shift(1)] = -volume.loc[close < close.shift(1)]
        return obv.cumsum()

    async def _rate_limit(self, api_name: str):
        """Async rate limiting with exponential backoff."""
        last_call = self.rate_limiter['last_call'].get(api_name, 0)
        time_since = time.time() - last_call
        
        if time_since < self.rate_limiter['min_interval']:
            await asyncio.sleep(self.rate_limiter['min_interval'] - time_since)
        
        self.rate_limiter['last_call'][api_name] = time.time()

    def _cache_key(self, prefix: str, **kwargs) -> str:
        return _cache._make_key(prefix, (), kwargs)

    def _cache_get(self, key: str):
        return _cache.get(key)

    def _cache_set(self, key: str, value: Any, ttl: int):
        _cache.set(key, value, ttl)

    def _notes_path(self) -> str:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(project_root, "notes.md")

    def _load_notes(self) -> List[Dict[str, str]]:
        path = self._notes_path()
        if not os.path.exists(path):
            return []
        try:
            with open(path, "r") as f:
                lines = [line.rstrip("\n") for line in f]
        except Exception:
            return []

        notes: List[Dict[str, str]] = []
        for line in lines:
            if not line.startswith("- "):
                continue
            payload = line[2:].strip()
            if " | " in payload:
                ts, note = payload.split(" | ", 1)
                notes.append({"timestamp": ts.strip(), "note": note.strip()})
            else:
                notes.append({"timestamp": "", "note": payload})

        if not notes and lines:
            raw = " ".join(line.strip() for line in lines if line.strip())
            if raw:
                notes.append({"timestamp": "", "note": raw})
        return notes

    def _write_notes(self, notes: List[Dict[str, str]]):
        path = self._notes_path()
        lines = ["# Notes", ""]
        for entry in notes:
            ts = entry.get("timestamp", "").strip()
            note = entry.get("note", "").strip()
            if ts:
                lines.append(f"- {ts} | {note}")
            else:
                lines.append(f"- {note}")
        lines.append("")
        with open(path, "w") as f:
            f.write("\n".join(lines))

    def _alpaca_list_positions(self):
        client = self.clients["alpaca"]
        if hasattr(client, "get_all_positions"):
            return client.get_all_positions()
        if hasattr(client, "list_positions"):
            return client.list_positions()
        raise AttributeError("Alpaca client missing positions list method")

    def _alpaca_list_orders(self, status: QueryOrderStatus = QueryOrderStatus.OPEN, nested: bool = True):
        client = self.clients["alpaca"]
        if hasattr(client, "get_orders"):
            order_filter = GetOrdersRequest(status=status, nested=nested)
            return client.get_orders(order_filter)
        if hasattr(client, "list_orders"):
            return client.list_orders()
        raise AttributeError("Alpaca client missing orders list method")

    def _alpaca_get_order(self, order_id):
        client = self.clients["alpaca"]
        if hasattr(client, "get_order_by_id"):
            return client.get_order_by_id(order_id)
        if hasattr(client, "get_order"):
            return client.get_order(order_id)
        raise AttributeError("Alpaca client missing get order method")

    def _alpaca_cancel_order(self, order_id):
        client = self.clients["alpaca"]
        if hasattr(client, "cancel_order_by_id"):
            return client.cancel_order_by_id(order_id)
        if hasattr(client, "cancel_order"):
            return client.cancel_order(order_id)
        raise AttributeError("Alpaca client missing cancel order method")

    def _alpaca_get_position(self, symbol):
        client = self.clients["alpaca"]
        if hasattr(client, "get_open_position"):
            return client.get_open_position(symbol)
        if hasattr(client, "get_position"):
            return client.get_position(symbol)
        raise AttributeError("Alpaca client missing get position method")
    
    def _log_tool_usage(self, tool_name: str, args: dict, result: any, execution_time: float = None):
        """Log tool usage with rotating buffer."""
        log_entry = {
            "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
            "tool_name": tool_name,
            "args": self._sanitize_for_log(args),
            "result": self._sanitize_for_log(result),
            "execution_time_ms": round(execution_time * 1000, 2) if execution_time else None
        }
        self.tool_usage_log.append(log_entry)
        
        if len(self.tool_usage_log) > 50:
            self.tool_usage_log = self.tool_usage_log[-50:]
    
    def _sanitize_for_log(self, obj):
        """Sanitize objects for logging."""
        if isinstance(obj, dict):
            sanitized = {}
            for k, v in obj.items():
                if isinstance(v, (str, int, float, bool, type(None))):
                    if isinstance(v, str) and len(v) > 200:
                        sanitized[k] = v[:200] + "... (truncated)"
                    else:
                        sanitized[k] = v
                elif isinstance(v, list) and len(v) > 5:
                    sanitized[k] = f"[List with {len(v)} items]"
                elif isinstance(v, dict):
                    sanitized[k] = "[Dict object]"
                else:
                    sanitized[k] = str(v)[:100] + "..." if len(str(v)) > 100 else str(v)
            return sanitized
        elif isinstance(obj, list):
            if len(obj) > 5:
                return f"[List with {len(obj)} items - first 2: {obj[:2]}]"
            else:
                return [self._sanitize_for_log(item) for item in obj]
        elif isinstance(obj, str) and len(obj) > 300:
            return obj[:300] + "... (truncated)"
        else:
            return obj
    
    @with_budget(max_tokens=30000)
    async def get_market_news(self) -> List[Dict]:
        """
        Async market news fetching.
        
        Returns:
            List of market news articles
        
        Example Tool Input:
            {
                "tool_name": "get_market_news",
                "args": {}
            }
        
        Example Output:
            [
                {
                    "category": "general",
                    "datetime": 1705344000,
                    "headline": "Fed Signals Potential Rate Cuts in 2024",
                    "id": 12345,
                    "image": "https://example.com/image.jpg",
                    "related": "AAPL,MSFT,GOOGL",
                    "source": "Reuters",
                    "summary": "Federal Reserve officials indicate...",
                    "url": "https://example.com/article"
                }
            ]
        """
        start_time = time.time()
        
        async with self.semaphores['finnhub']:
            await self._rate_limit('finnhub')
            
            try:
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None, 
                    lambda: requests.get(
                        "https://finnhub.io/api/v1/news",
                        params={"category": "general", "token": self.clients["finnhub_key"]},
                        timeout=10
                    )
                )
                
                result = response.json()[:25] if response.status_code == 200 else []
                self._log_tool_usage("get_market_news", {}, result, time.time() - start_time)
                return result
                
            except Exception as e:
                logging.error(f"Error fetching news: {e}")
                result = []
                self._log_tool_usage("get_market_news", {}, {"error": str(e)}, time.time() - start_time)
                return result
    
    @with_budget(max_tokens=40000)
    async def get_latest_sec_filings(self, symbol: str) -> List[Dict]:
        """
        Async SEC filings retrieval with raw text and robust error handling.
        
        Args:
            symbol: Stock symbol to get filings for
        
        Returns:
            List of SEC filings with raw text
        
        Example Tool Input:
            {
                "tool_name": "get_latest_sec_filings",
                "args": {"symbol": "AAPL"}
            }
        
        Example Output:
            [
                {
                    "filedAt": "2024-01-15T16:30:00Z",
                    "formType": "10-K",
                    "link": "https://www.sec.gov/Archives/edgar/data/...",
                    "linkToTxt": "https://www.sec.gov/Archives/edgar/data/...",
                    "cik": "0000320193",
                    "rawText": "UNITED STATES SECURITIES AND EXCHANGE COMMISSION...",
                    "summary": "Annual report for fiscal year ending..."
                }
            ]
        """
        if not self.clients["queryapi"]:
            return [{"error": "SEC API not configured"}]
        start_time = time.time()
        async with self.semaphores['sec']:
            await self._rate_limit('sec')
            query = {
                "query": {"query_string": {"query": f"ticker:{symbol}"}},
                "from": "0", "size": "5",
                "sort": [{"filedAt": {"order": "desc"}}]
            }
            try:
                loop = asyncio.get_event_loop()
                filings = await loop.run_in_executor(
                    None, 
                    lambda: self.clients["queryapi"].get_filings(query)
                )
                if not filings or 'filings' not in filings:
                    logging.warning(f"No filings returned for {symbol} from SEC API: {filings}")
                    return [{"error": "No filings returned from SEC API"}]
                results = []
                for f in filings.get('filings', []):
                    filedAt = f.get('filedAt', '')
                    formType = f.get('formType', '')
                    link = f.get('linkToFilingDetails') or f.get('linkToHtml', '')
                    link_to_txt = f.get('linkToTxt', '')
                    cik = f.get('cik', '')
                    raw_text = None
                    if cik and link_to_txt:
                        try:
                            headers = {
                                "User-Agent": "Magnus Ohle (magnus.ohle@gmail.com)",
                                "Accept-Encoding": "gzip, deflate"
                            }
                            # Use async requests for text fetching
                            import aiohttp
                            async with aiohttp.ClientSession() as session:
                                async with session.get(link_to_txt, headers=headers, timeout=10) as resp:
                                    if resp.status == 200:
                                        raw_text = await resp.text()
                                    else:
                                        raw_text = f"HTTP {resp.status}: {resp.reason}"
                        except Exception as e:
                            raw_text = f"[Error fetching text: {e}]"
                    else:
                        raw_text = "[No CIK or linkToTxt available]"
                    results.append({
                        "filedAt": filedAt,
                        "formType": formType,
                        "link": link,
                        "linkToTxt": link_to_txt,
                        "cik": cik,
                        "rawText": raw_text[:2000] + ("..." if raw_text and len(raw_text) > 2000 else "")
                    })
                self._log_tool_usage("get_latest_sec_filings", {"symbol": symbol}, results, time.time() - start_time)
                return results
            except Exception as e:
                logging.warning(f"SEC filings retrieval error for {symbol}: {e}")
                return [{"error": f"Could not retrieve filings: {e}"}]
    
    @with_budget(max_tokens=40000)
    async def analyze_technicals(self, symbol: str) -> Dict:
        """
        Async technical analysis with robust logic and config-driven thresholds.
        """
        import time
        import numpy as np
        import pandas as pd
        import random
        import asyncio
        start_time = time.time()
        args = {"symbol": symbol}
        symbol = symbol.upper().strip()
        # Add a short random sleep to avoid burst requests
        await asyncio.sleep(random.uniform(0.2, 1.0))
        try:
            async with asyncio.timeout(30):  # Elongated timeout for slow data fetches
                async with self.semaphores['yfinance']:
                    loop = asyncio.get_event_loop()
                    # Retry logic for transient errors
                    max_retries = 3
                    for attempt in range(max_retries):
                        try:
                            df = await loop.run_in_executor(
                                None,
                                lambda: yf.download(
                                    tickers=symbol,
                                    period="300d",
                                    interval="1d",
                                    progress=False,
                                    auto_adjust=False
                                )
                            )
                            break
                        except Exception as e:
                            if attempt == max_retries - 1:
                                raise
                            await asyncio.sleep(1.5 * (attempt + 1))  # Exponential backoff
                    # --- FIX: Flatten MultiIndex columns if present ---
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = ['_'.join([str(i) for i in col]).strip('_') for col in df.columns.values]
                        close_col = f'Close_{symbol}'
                        high_col = f'High_{symbol}'
                        low_col = f'Low_{symbol}'
                        open_col = f'Open_{symbol}'
                        volume_col = f'Volume_{symbol}'
                    else:
                        close_col = get_close_col(df, symbol)
                        high_col = 'High'
                        low_col = 'Low'
                        open_col = 'Open'
                        volume_col = 'Volume'
                    
                    # Check for explicit yahoo errors
                    if symbol in yf_shared._ERRORS:
                         error_msg = f"Yahoo Finance Error for {symbol}: {yf_shared._ERRORS[symbol]}"
                         self._log_tool_usage("analyze_technicals", args, {"error": error_msg}, time.time() - start_time)
                         return {"error": error_msg}

                    if df.empty or len(df) < 200 or close_col not in df.columns:
                        result = {"error": f"Insufficient data for {symbol} (rows={len(df)}). Check symbol validity or delisting status."}
                        self._log_tool_usage("analyze_technicals", args, result, time.time() - start_time)
                        return result
                    df['RSI'] = self._calculate_rsi(df[close_col], length=14)
                    macd_data = self._calculate_macd(df[close_col])
                    if macd_data is None or not isinstance(macd_data, pd.DataFrame) or macd_data.isna().all().all():
                        result = {"error": "MACD calculation failed"}
                        self._log_tool_usage("analyze_technicals", args, result, time.time() - start_time)
                        return result
                    df['MACD'] = macd_data['MACD']
                    df['MACD_signal'] = macd_data['MACDs']
                    df['ATR'] = self._calculate_atr(df[high_col], df[low_col], df[close_col])
                    df['SMA_20'] = df[close_col].rolling(20).mean()
                    df['SMA_50'] = df[close_col].rolling(50).mean()
                    df['SMA_200'] = df[close_col].rolling(200).mean()
                    df['Volume_SMA'] = df[volume_col].rolling(20).mean()
                    def find_pivot_points(highs, lows, window=5):
                        pivots_high, pivots_low = [], []
                        for i in range(window, len(highs) - window):
                            if all(highs[i] >= highs[j] for j in range(i-window, i)):
                                pivots_high.append(highs[i])
                            if all(lows[i] <= lows[j] for j in range(i-window, i)):
                                pivots_low.append(lows[i])
                        return pivots_high, pivots_low
                    pivots_high, pivots_low = find_pivot_points(df[high_col].values, df[low_col].values)
                    recent_high = max(pivots_high[-3:]) if len(pivots_high) >= 3 else df[high_col].tail(20).max()
                    recent_low  = min(pivots_low[-3:])  if len(pivots_low)  >= 3 else df[low_col].tail(20).min()
                    current_price = df[close_col].iloc[-1]
                    psychological = [lvl for lvl in [10, 25, 50, 100, 150, 200, 250, 300, 500, 1000]
                                     if current_price * 0.95 <= lvl <= current_price * 1.05]
                    resistance = max([recent_high] + psychological) if psychological else recent_high
                    support    = recent_low
                    rsi_val = df['RSI'].iloc[-1] if 'RSI' in df and not pd.isna(df['RSI'].iloc[-1]) else 50.0
                    macd_val = df['MACD'].iloc[-1] if 'MACD' in df and not pd.isna(df['MACD'].iloc[-1]) else 0.0
                    sig_val  = df['MACD_signal'].iloc[-1] if 'MACD_signal' in df and not pd.isna(df['MACD_signal'].iloc[-1]) else 0.0
                    macd_signal = 'bullish' if macd_val > sig_val else 'bearish'
                    sma50 = df['SMA_50'].iloc[-1] if 'SMA_50' in df and not pd.isna(df['SMA_50'].iloc[-1]) else current_price
                    sma200 = df['SMA_200'].iloc[-1] if 'SMA_200' in df and not pd.isna(df['SMA_200'].iloc[-1]) else current_price
                    if current_price > sma50 and sma50 > sma200:
                        trend = 'uptrend'
                    elif current_price < sma50 and sma50 < sma200:
                        trend = 'downtrend'
                    else:
                        trend = 'sideways'
                    current_volume = df[volume_col].iloc[-1]
                    vol_sma_20 = df['Volume_SMA'].iloc[-1] if 'Volume_SMA' in df and not pd.isna(df['Volume_SMA'].iloc[-1]) else current_volume
                    vol_sma_50 = df[volume_col].rolling(50).mean().iloc[-1] if len(df) >= 50 else current_volume
                    volume_ratio_20 = current_volume / vol_sma_20 if vol_sma_20 > 0 else 1.0
                    volume_ratio_50 = current_volume / vol_sma_50 if vol_sma_50 > 0 else 1.0
                    recent_price_change = (current_price - df[close_col].iloc[-5]) / df[close_col].iloc[-5] if len(df) >= 5 else 0
                    recent_volume_avg = df[volume_col].tail(5).mean()
                    volume_on_up_days = df[df[close_col] > df[close_col].shift(1)][volume_col].tail(10).mean() if len(df) >= 10 else current_volume
                    volume_on_down_days = df[df[close_col] < df[close_col].shift(1)][volume_col].tail(10).mean() if len(df) >= 10 else current_volume
                    if volume_ratio_20 > CONFIG["volume_surge_ratio_20"] and volume_ratio_50 > CONFIG["volume_surge_ratio_50"]:
                        volume_trend = 'surge'
                    elif volume_ratio_20 > CONFIG["volume_increasing_ratio_20"] and volume_ratio_50 > CONFIG["volume_increasing_ratio_50"]:
                        volume_trend = 'increasing'
                    elif volume_ratio_20 < CONFIG["volume_declining_ratio_20"] and volume_ratio_50 < CONFIG["volume_declining_ratio_50"]:
                        volume_trend = 'declining'
                    elif volume_ratio_20 < CONFIG["volume_drying_up_ratio_20"]:
                        volume_trend = 'drying_up'
                    else:
                        volume_trend = 'stable'
                    volume_confirms_trend = False
                    if trend == 'uptrend' and volume_on_up_days > volume_on_down_days * 1.2:
                        volume_confirms_trend = True
                    elif trend == 'downtrend' and volume_on_down_days > volume_on_up_days * 1.2:
                        volume_confirms_trend = True
                    atr_val = df['ATR'].iloc[-1] if 'ATR' in df and not pd.isna(df['ATR'].iloc[-1]) else current_price * 0.02
                    result = {
                        "symbol": symbol,
                        "price": float(current_price),
                        "rsi": float(rsi_val),
                        "macd_signal": macd_signal,
                        "trend": trend,
                        "volatility": float(atr_val / current_price),
                        "support": float(support),
                        "resistance": float(resistance),
                        "volume_trend": volume_trend,
                        "volume_ratio_20d": float(volume_ratio_20),
                        "volume_ratio_50d": float(volume_ratio_50),
                        "volume_confirms_trend": volume_confirms_trend,
                        "current_volume": int(current_volume),
                        "avg_volume_20d": int(vol_sma_20),
                        "open_col": open_col,
                        "recent_price_change_pct": float(recent_price_change * 100),
                        "recent_volume_avg": int(recent_volume_avg),
                        "volume_on_up_days": int(volume_on_up_days),
                        "volume_on_down_days": int(volume_on_down_days),
                        "macd_value": float(macd_val),
                        "macd_signal_value": float(sig_val),
                        "sma_50": float(sma50),
                        "sma_200": float(sma200)
                    }
                    self._log_tool_usage("analyze_technicals", args, result, time.time() - start_time)
                    return result
        except Exception as e:
            logging.error(f"Error in technical analysis for {symbol}: {e}")
            result = {"error": str(e)}
            self._log_tool_usage("analyze_technicals", args, result, time.time() - start_time)
            return result
    
    async def assess_market_regime(self) -> dict:
        """
        Async market regime assessment using a small basket of representative ETFs for speed.
        """
        import numpy as np
        import pandas as pd
        import yfinance as yf
        import time
        start_time = time.time()
        args = {}
        try:
            loop = asyncio.get_event_loop()
            logging.info("[REGIME] Assessing market regime with lightweight ETF basket.")

            # --- Step 1: Define a small basket of representative ETFs ---
            representative_etfs = ["SPY", "QQQ", "IWM", "DIA", "XLF", "XLK"]
            all_tickers = ["^VIX"] + representative_etfs

            # --- Step 2: Download data for the small basket ---
            data = await loop.run_in_executor(
                self.executor,
                lambda: yf.download(
                    tickers=all_tickers,
                    period="250d",
                    interval="1d",
                    progress=False,
                    auto_adjust=True, # Use auto_adjust for simplicity
                    group_by='ticker'
                )
            )

            if data.empty:
                raise ValueError("yf.download returned an empty DataFrame.")

            logging.info(f"[REGIME] yf.download data columns: {data.columns}")

            # --- Step 3: Robustly get SPY close series ---
            spy_close = None
            if 'SPY' in data:
                # Multi-ticker, multi-index
                if 'Close' in data['SPY']:
                    spy_close = data['SPY']['Close']
            elif ('Close', 'SPY') in data.columns:
                spy_close = data[('Close', 'SPY')]
            elif 'Close' in data:
                # Single ticker, single-index
                spy_close = data['Close']
            if spy_close is None or spy_close.empty:
                logging.error(f"SPY close price not found. Data columns: {data.columns}")
                raise ValueError(f"SPY close price not found in yfinance data. Columns: {data.columns}")
            
            logging.info(f"[REGIME] SPY Close series head: {spy_close.head().to_string()}")

            sma_50 = spy_close.rolling(50).mean().iloc[-1]
            sma_200 = spy_close.rolling(200).mean().iloc[-1]

            if sma_50 > sma_200:
                trend = "bull"
            elif sma_50 < sma_200:
                trend = "bear"
            else:
                trend = "sideways"

            # --- Step 4: Get VIX Level ---
            vix_level = None
            if '^VIX' in data:
                vix_close = data['^VIX']['Close'] if 'Close' in data['^VIX'] else None
                if vix_close is not None and not vix_close.empty:
                    vix_level = vix_close.iloc[-1]
            if vix_level is None:
                vix_level = 20.0
            vix_low = 15
            vix_high = 25

            if vix_level < vix_low:
                volatility = "low"
            elif vix_level < vix_high:
                volatility = "normal"
            else:
                volatility = "high"

            # --- Step 5: Calculate Market Breadth from ETF basket ---
            above_50_sma_count = 0
            for ticker in representative_etfs:
                ticker_close = None
                if ticker in data:
                    if 'Close' in data[ticker]:
                        ticker_close = data[ticker]['Close']
                elif ('Close', ticker) in data.columns:
                    ticker_close = data[('Close', ticker)]
                elif 'Close' in data:
                    ticker_close = data['Close']
                if ticker_close is not None and not ticker_close.empty and len(ticker_close) >= 50:
                    try:
                        ticker_sma_50 = ticker_close.rolling(50).mean().iloc[-1]
                        if ticker_close.iloc[-1] > ticker_sma_50:
                            above_50_sma_count += 1
                    except (IndexError, KeyError):
                        continue # Skip if data is incomplete

            market_breadth_pct = (above_50_sma_count / len(representative_etfs)) * 100 if representative_etfs else 50.0

            # --- Step 6: Synthesize Result ---
            should_be_defensive = trend == "bear" or volatility == "high" or market_breadth_pct < 40

            result = {
                "trend": trend,
                "volatility": volatility,
                "vix_level": round(vix_level, 2),
                "market_breadth_pct": round(market_breadth_pct, 2),
                "should_be_defensive": should_be_defensive,
                "analysis_type": "lightweight_etf_basket"
            }
            self._log_tool_usage("assess_market_regime", args, result, time.time() - start_time)
            return result

        except Exception as e:
            logging.error(f"Lightweight market regime error: {e}", exc_info=True)
            result = {
                "trend": "unknown", "volatility": "normal",
                "vix_level": 20.0,
                "market_breadth_pct": 50.0, "should_be_defensive": True,
                "error": str(e)
            }
            self._log_tool_usage("assess_market_regime", args, result, time.time() - start_time)
            return result
    
    async def run_plan(self, plan_parts: List[Dict]) -> List[Any]:
        """Async plan execution with clear separators for each tool result."""
        results = []
        for part in plan_parts:
            tool_name = part.get('tool_name') or part.get('name')
            args = part.get('args', {})
            sep = f"\n===== TOOL: {tool_name} =====\n"
            end_sep = f"\n===== END TOOL: {tool_name} =====\n"
            if tool_name and hasattr(self, tool_name):
                tool_func = getattr(self, tool_name)
                try:
                    if asyncio.iscoroutinefunction(tool_func):
                        result = await tool_func(**args)
                    else:
                        loop = asyncio.get_event_loop()
                        result = await loop.run_in_executor(None, lambda: tool_func(**args))
                    # Add separators to result for clarity
                    results.append(f"{sep}{result}{end_sep}")
                except Exception as e:
                    results.append(f"{sep}ERROR: {str(e)}{end_sep}")
            else:
                results.append(f"{sep}ERROR: Tool '{tool_name}' not found{end_sep}")
        return results
    
    # Watcher methods
    async def check_technical(self, symbol: str) -> bool:
        """Check technical conditions for a symbol."""
        if self.watcher_manager:
            return await self.watcher_manager.check_technical_async(symbol)
        return False
    
    def register_watcher(self, symbol: str, interval: float = 60.0, condition=None, callback=None):
        """Register a technical watcher."""
        if self.watcher_manager:
            return self.watcher_manager.register_watcher(symbol, interval, condition, callback)
        return None
    
    def cancel_watcher(self, symbol: str) -> bool:
        """Cancel a watcher."""
        if self.watcher_manager:
            return self.watcher_manager.cancel_watcher(symbol)
        return False
    
    def list_watchers(self) -> Dict[str, Dict]:
        """List all active watchers."""
        if self.watcher_manager:
            return self.watcher_manager.list_watchers()
        return {}
    
    async def start_watchers(self):
        """Start all registered watchers."""
        if self.watcher_manager:
            await self.watcher_manager.start_all_watchers()
    
    async def stop_watchers(self):
        """Stop all watchers."""
        if self.watcher_manager:
            await self.watcher_manager.stop_all_watchers()
    
    # Scanner methods
    async def scan_top_movers(self, limit: int = 50) -> List[Dict]:
        """
        Scan for top market movers.
        
        Args:
            limit: Maximum number of results to return
        
        Returns:
            List of top market movers
        
        Example Tool Input:
            {
                "tool_name": "scan_top_movers",
                "args": {"limit": 20}
            }
        
        Example Output:
            [
                {
                    "symbol": "NVDA",
                    "change_pct": 5.2,
                    "volume": 45678900,
                    "price": 485.50
                }
            ]
        """
        if self.market_scanner:
            return await self.market_scanner.scan_top_movers(limit)
        return []
    
    async def scan_unusual_volume(self, threshold: float = 2.0, limit: int = 50) -> List[Dict]:
        """
        Scan for unusual volume.
        
        Args:
            threshold: Volume threshold multiplier
            limit: Maximum number of results to return
        
        Returns:
            List of stocks with unusual volume
        
        Example Tool Input:
            {
                "tool_name": "scan_unusual_volume",
                "args": {"threshold": 2.5, "limit": 20}
            }
        
        Example Output:
            [
                {
                    "symbol": "TSLA",
                    "volume_ratio": 3.2,
                    "price": 245.75,
                    "volume": 125000000
                }
            ]
        """
        if self.market_scanner:
            return await self.market_scanner.scan_unusual_volume(threshold, limit)
        return []

    async def scan_unusual_options_flow(
        self,
        symbols: List[str],
        min_volume: int = 500,
        min_volume_oi_ratio: float = 2.0,
        max_expirations: int = 3,
        max_days_out: int = 45,
        limit: int = 50,
    ) -> List[Dict]:
        """
        Scan for unusual options activity using volume vs open interest.
        """
        import time
        start_time = time.time()
        args = {
            "symbols": symbols,
            "min_volume": min_volume,
            "min_volume_oi_ratio": min_volume_oi_ratio,
            "max_expirations": max_expirations,
            "max_days_out": max_days_out,
            "limit": limit,
        }
        results: List[Dict] = []
        if not symbols:
            self._log_tool_usage("scan_unusual_options_flow", args, results, time.time() - start_time)
            return results

        loop = asyncio.get_running_loop()
        today = dt.date.today()

        for symbol in symbols:
            ticker = yf.Ticker(symbol)
            try:
                expirations = await loop.run_in_executor(None, lambda: ticker.options)
            except Exception as e:
                logging.warning(f"Options expiration fetch failed for {symbol}: {e}")
                continue

            if not expirations:
                continue

            for exp in expirations[:max_expirations]:
                try:
                    exp_date = dt.date.fromisoformat(exp)
                except Exception:
                    continue
                if max_days_out is not None and (exp_date - today).days > max_days_out:
                    continue

                try:
                    chain = await loop.run_in_executor(None, lambda: ticker.option_chain(exp))
                except Exception as e:
                    logging.warning(f"Options chain fetch failed for {symbol} {exp}: {e}")
                    continue

                for side, df in [("call", chain.calls), ("put", chain.puts)]:
                    if df is None or df.empty:
                        continue
                    df = df.copy()
                    if "volume" not in df or "openInterest" not in df:
                        continue
                    vol = df["volume"].fillna(0)
                    oi = df["openInterest"].fillna(0)
                    mask = (vol >= min_volume) & (oi > 0)
                    df = df.loc[mask].copy()
                    if df.empty:
                        continue
                    df["volume_oi_ratio"] = vol.loc[df.index] / oi.loc[df.index]
                    df = df[df["volume_oi_ratio"] >= min_volume_oi_ratio]
                    if df.empty:
                        continue

                    for row in df.itertuples():
                        results.append(
                            {
                                "symbol": symbol,
                                "contract_type": side,
                                "expiration": exp,
                                "strike": float(getattr(row, "strike", 0.0)),
                                "last_price": float(getattr(row, "lastPrice", 0.0)),
                                "volume": int(getattr(row, "volume", 0) or 0),
                                "open_interest": int(getattr(row, "openInterest", 0) or 0),
                                "volume_oi_ratio": float(getattr(row, "volume_oi_ratio", 0.0)),
                                "implied_volatility": float(getattr(row, "impliedVolatility", 0.0) or 0.0),
                            }
                        )

        results.sort(key=lambda x: (x.get("volume_oi_ratio", 0), x.get("volume", 0)), reverse=True)
        results = results[:limit]
        self._log_tool_usage("scan_unusual_options_flow", args, results, time.time() - start_time)
        return results

    async def scan_dark_pool_activity(
        self,
        symbol: str = None,
        limit: int = 25,
        report_type: str = "NMS",
        force_refresh: bool = False,
    ) -> Dict:
        """
        Scan FINRA ATS transparency data (weekly, delayed) for dark pool activity.
        """
        import time

        start_time = time.time()
        args = {
            "symbol": symbol,
            "limit": limit,
            "report_type": report_type,
            "force_refresh": force_refresh,
        }

        result = {
            "error": "Disabled: FINRA ATS weekly data is blocked/paid. Enable when a licensed source is available."
        }
        self._log_tool_usage("scan_dark_pool_activity", args, result, time.time() - start_time)
        return result

    async def scan_insider_activity(
        self,
        symbol: str = None,
        limit: int = 25,
        only_buys: bool = True,
        force_refresh: bool = False,
    ) -> Dict:
        """
        Scan insider transactions via EarningsFeed API (free tier).
        """
        import time
        start_time = time.time()
        args = {
            "symbol": symbol,
            "limit": limit,
            "only_buys": only_buys,
            "force_refresh": force_refresh,
        }

        cache_key = self._cache_key("scan_insider_activity", **args)
        if not force_refresh:
            cached = self._cache_get(cache_key)
            if cached is not None:
                self._log_tool_usage("scan_insider_activity", args, cached, time.time() - start_time)
                return cached

        api_key = os.getenv("EARNINGSFEED_API_KEY")
        if not api_key:
            result = {"error": "EARNINGSFEED_API_KEY not set"}
            self._log_tool_usage("scan_insider_activity", args, result, time.time() - start_time)
            return result

        params = {"limit": limit}
        if symbol:
            params["ticker"] = symbol.upper()

        try:
            response = requests.get(
                "https://earningsfeed.com/api/v1/insider/transactions",
                headers={"Authorization": f"Bearer {api_key}"},
                params=params,
                timeout=15,
            )
            if response.status_code != 200:
                result = {"error": f"EarningsFeed response {response.status_code}"}
                self._log_tool_usage("scan_insider_activity", args, result, time.time() - start_time)
                return result

            payload = response.json()
            items = payload.get("items", payload)
            if only_buys:
                buy_codes = {"P", "A", "L"}
                filtered = []
                for item in items:
                    code = (item.get("transactionCode") or item.get("code") or "").upper()
                    if code in buy_codes:
                        filtered.append(item)
                items = filtered

            result = {"items": items[:limit], "source": "EarningsFeed"}
            self._cache_set(cache_key, result, ttl=24 * 3600)
            self._log_tool_usage("scan_insider_activity", args, result, time.time() - start_time)
            return result
        except Exception as e:
            result = {"error": str(e)}
            self._log_tool_usage("scan_insider_activity", args, result, time.time() - start_time)
            return result

    async def scan_13f_changes(
        self,
        symbol: str = None,
        limit: int = 25,
        force_refresh: bool = False,
    ) -> Dict:
        """
        Scan institutional 13F holdings via EarningsFeed API (free tier).
        """
        import time
        start_time = time.time()
        args = {"symbol": symbol, "limit": limit, "force_refresh": force_refresh}

        cache_key = self._cache_key("scan_13f_changes", **args)
        if not force_refresh:
            cached = self._cache_get(cache_key)
            if cached is not None:
                self._log_tool_usage("scan_13f_changes", args, cached, time.time() - start_time)
                return cached

        api_key = os.getenv("EARNINGSFEED_API_KEY")
        if not api_key:
            result = {"error": "EARNINGSFEED_API_KEY not set"}
            self._log_tool_usage("scan_13f_changes", args, result, time.time() - start_time)
            return result

        params = {"limit": limit}
        if symbol:
            params["ticker"] = symbol.upper()

        try:
            response = requests.get(
                "https://earningsfeed.com/api/v1/institutional/holdings",
                headers={"Authorization": f"Bearer {api_key}"},
                params=params,
                timeout=20,
            )
            if response.status_code != 200:
                result = {"error": f"EarningsFeed response {response.status_code}"}
                self._log_tool_usage("scan_13f_changes", args, result, time.time() - start_time)
                return result

            payload = response.json()
            items = payload.get("items", payload)
            result = {"items": items[:limit], "source": "EarningsFeed"}
            self._cache_set(cache_key, result, ttl=7 * 24 * 3600)
            self._log_tool_usage("scan_13f_changes", args, result, time.time() - start_time)
            return result
        except Exception as e:
            result = {"error": str(e)}
            self._log_tool_usage("scan_13f_changes", args, result, time.time() - start_time)
            return result
    
    async def scan_premarket_movers(self, limit: int = 20) -> List[Dict]:
        """
        Scan premarket movers.
        
        Args:
            limit: Maximum number of results to return
        
        Returns:
            List of premarket movers
        
        Example Tool Input:
            {
                "tool_name": "scan_premarket_movers",
                "args": {"limit": 15}
            }
        
        Example Output:
            [
                {
                    "symbol": "AAPL",
                    "premarket_change_pct": 2.1,
                    "volume": 1250000,
                    "price": 185.50
                }
            ]
        """
        if self.market_scanner:
            return await self.market_scanner.scan_premarket_movers(limit)
        return []
    
    async def get_market_summary(self) -> Dict:
        """
        Get market summary.
        
        Returns:
            Market summary data
        
        Example Tool Input:
            {
                "tool_name": "get_market_summary",
                "args": {}
            }
        
        Example Output:
            {
                "spy_change_pct": 0.5,
                "vix_level": 18.5,
                "advancing_stocks": 2456,
                "declining_stocks": 1234
            }
        """
        import time
        start_time = time.time()
        args = {}
        try:
            # Fetch market data
            spy_df = yf.download('SPY', period='1d', interval='1m')
            vix_df = yf.download('VIX', period='1d', interval='1m')
            
            if spy_df.empty or vix_df.empty:
                result = {"error": "Failed to fetch market data"}
                self._log_tool_usage("get_market_summary", args, result, time.time() - start_time)
                return result
            
            spy_change_pct = spy_df['Close'].pct_change().iloc[-1] * 100
            vix_level = vix_df['Close'].iloc[-1]
            
            # Fetch additional data
            adv_stocks = 0
            decl_stocks = 0
            if self.market_scanner:
                movers = await self.market_scanner.scan_top_movers(limit=500)
                if movers:
                    adv_stocks = sum(1 for m in movers if m.get("change_pct", 0) > 0)
                    decl_stocks = sum(1 for m in movers if m.get("change_pct", 0) < 0)
            
            result = {
                "spy_change_pct": spy_change_pct,
                "vix_level": vix_level,
                "advancing_stocks": adv_stocks,
                "declining_stocks": decl_stocks
            }
            self._log_tool_usage("get_market_summary", args, result, time.time() - start_time)
            return result
        except Exception as e:
            logging.error(f"Error fetching market summary: {e}")
            result = {"error": str(e)}
            self._log_tool_usage("get_market_summary", args, result, time.time() - start_time)
            return result
    
    # Calendar methods
    async def get_earnings_calendar(self, start_date=None, end_date=None, limit=100) -> List[Dict]:
        """
        Get earnings calendar.
        
        Args:
            start_date: Start date for calendar
            end_date: End date for calendar
            limit: Maximum number of results
        
        Returns:
            List of earnings events
        
        Example Tool Input:
            {
                "tool_name": "get_earnings_calendar",
                "args": {"limit": 50}
            }
        
        Example Output:
            [
                {
                    "symbol": "AAPL",
                    "date": "2024-01-25",
                    "time": "16:30",
                    "estimate": 2.10
                }
            ]
        """
        if self.earnings_calendar:
            return await self.earnings_calendar.get_earnings_calendar(start_date, end_date, limit)
        return []
    
    async def get_earnings_for_symbols(self, symbols: List[str]) -> List[Dict]:
        """
        Get earnings for specific symbols.
        
        Args:
            symbols: List of stock symbols
        
        Returns:
            List of earnings data for symbols
        
        Example Tool Input:
            {
                "tool_name": "get_earnings_for_symbols",
                "args": {"symbols": ["AAPL", "MSFT", "GOOGL"]}
            }
        
        Example Output:
            [
                {
                    "symbol": "AAPL",
                    "next_earnings": "2024-01-25",
                    "estimate": 2.10,
                    "previous": 1.85
                }
            ]
        """
        if self.earnings_calendar:
            return await self.earnings_calendar.get_earnings_for_symbols(symbols)
        return []
    
    async def get_earnings_surprises(self, symbols: List[str]) -> List[Dict]:
        """
        Get earnings surprises.
        
        Args:
            symbols: List of stock symbols
        
        Returns:
            List of earnings surprises
        
        Example Tool Input:
            {
                "tool_name": "get_earnings_surprises",
                "args": {"symbols": ["AAPL", "MSFT", "GOOGL"]}
            }
        
        Example Output:
            [
                {
                    "symbol": "AAPL",
                    "surprise_pct": 5.2,
                    "actual": 2.15,
                    "estimate": 2.04
                }
            ]
        """
        if self.earnings_calendar:
            return await self.earnings_calendar.get_earnings_surprises(symbols)
        return []
    
    # Utility methods
    def validate_order(self, order_data: Dict) -> Dict[str, Any]:
        """Validate smart order entry data."""
        return validate_smart_order(order_data)
    
    def dedupe_content(self, text: str) -> Dict[str, Any]:
        """Check for duplicate content."""
        return dedupe_blob(text)
    
    def rotate_logs(self) -> Dict[str, Any]:
        """Rotate tool usage logs."""
        return rotate_tool_usage_log(self.tool_usage_log)
    
    def validate_data(self, data: Any, schema: Dict) -> Dict[str, Any]:
        """Validate data against schema."""
        return strict_json_schema_validate(data, schema)
    
    def get_recent_tool_usage(self, limit: int = 10) -> List[Dict]:
        """Get recent tool usage entries."""
        return self.tool_usage_log[-limit:] if self.tool_usage_log else []

    async def calculate_position_size(self, symbol: str, confidence: str, entry_price: float, force_minimum: bool = True, trade_history: List[Dict] = None):
        """
        Async Kelly Criterion-based position sizing with constraints, using trade history, volume, and sector exposure.
        
        Args:
            symbol: Stock symbol
            confidence: Confidence level ("high", "medium", "low")
            entry_price: Entry price for the trade
            force_minimum: Whether to force minimum position size
            trade_history: Optional list of trade history for win rate calculation
        
        Returns:
            Tuple of (shares, position_details)
        
        Example Tool Input:
            {
                "tool_name": "calculate_position_size",
                "args": {
                    "symbol": "NVDA",
                    "confidence": "high",
                    "entry_price": 485.50,
                    "force_minimum": true
                }
            }
        
        Example Output:
            (25, {
                "position_size_pct": 2.5,
                "kelly_fraction": 0.15,
                "position_value": 12137.50,
                "volume_adjustment": "increased_25pct_strong_volume"
            })
        """
        import time
        start_time = time.time()
        args = {"symbol": symbol, "confidence": confidence, "entry_price": entry_price, "force_minimum": force_minimum}
        try:
            loop = asyncio.get_running_loop()
            account = await loop.run_in_executor(self.executor, self.clients["alpaca"].get_account)
            buying_power = float(account.buying_power)
            total_equity = float(account.equity)
            
            # Use provided trade history or fall back to default values
            trades = trade_history or []
            if len(trades) >= 10:
                winning_trades = [t for t in trades if t.get('profit', 0) > 0]
                losing_trades = [t for t in trades if t.get('profit', 0) < 0]
                win_rate = len(winning_trades) / len(trades)
                avg_win = np.mean([t['profit'] for t in winning_trades]) if winning_trades else 0.08
                avg_loss = abs(np.mean([t['profit'] for t in losing_trades])) if losing_trades else 0.04
                symbol_trades = [t for t in trades if t.get('symbol') == symbol]
                if len(symbol_trades) >= 5:
                    symbol_winners = [t for t in symbol_trades if t.get('profit', 0) > 0]
                    symbol_win_rate = len(symbol_winners) / len(symbol_trades)
                    win_rate = (win_rate * 0.7) + (symbol_win_rate * 0.3)
            else:
                win_rate = 0.52
                avg_win = 0.06
                avg_loss = 0.03
                
            confidence_map = {"high": 0.025, "medium": 0.015, "low": 0.01}
            base_size = confidence_map.get(confidence.lower(), 0.01)
            
            # Volume-adjusted position sizing
            volume_adjustment = "no_volume_analysis"
            try:
                volume_analysis = await self.analyze_volume_trend(symbol)
                if not volume_analysis.get('error'):
                    volume_quality = volume_analysis.get('volume_quality_score', 50)
                    volume_pattern = volume_analysis.get('volume_pattern', 'normal')
                    if volume_quality >= 70 and volume_pattern in ['explosive_surge', 'strong_increase']:
                        base_size *= CONFIG.get("volume_adjustment_high", 1.25)
                        volume_adjustment = "increased_25pct_strong_volume"
                    elif volume_quality >= 50 and volume_pattern in ['moderate_increase']:
                        base_size *= CONFIG.get("volume_adjustment_medium", 1.1)
                        volume_adjustment = "increased_10pct_good_volume"
                    elif volume_pattern in ['declining', 'severely_dry']:
                        base_size *= CONFIG.get("volume_adjustment_low", 0.7)
                        volume_adjustment = "reduced_30pct_weak_volume"
                    else:
                        volume_adjustment = "no_volume_adjustment"
                else:
                    volume_adjustment = "volume_analysis_failed"
            except Exception as e:
                logging.warning(f"Volume analysis failed for position sizing: {e}")
                volume_adjustment = "volume_analysis_error"
                
            if avg_loss > 0:
                kelly_fraction = (win_rate * (avg_win/avg_loss) - (1-win_rate)) / (avg_win/avg_loss)
            else:
                kelly_fraction = 0.01
            kelly_fraction = max(0, min(kelly_fraction, 0.25))
            position_size = base_size * (1 + kelly_fraction)
            max_position_size = CONFIG['max_position_size_pct'] / 100
            position_size = min(position_size, max_position_size)
            
            sector = SECTOR_MAP.get(symbol, "Other")
            sector_exposure = await self.analyze_sector_exposure()
            current_sector_pct = sector_exposure['sectors'].get(sector, 0)
            if current_sector_pct > CONFIG['max_sector_exposure_pct'] * 0.8:
                position_size *= 0.5
                
            # === LIQUIDITY CONSTRAINT ===
            liquidity_adjustment_reason = "none"
            try:
                # We can reuse the volume analysis we already performed
                if volume_analysis and not volume_analysis.get('error'):
                    avg_daily_volume = volume_analysis.get('avg_volume_20d', 0)
                    if avg_daily_volume > 0:
                        max_participation_rate = CONFIG.get('max_trade_participation_rate', 0.025)
                        max_shares_by_liquidity = avg_daily_volume * max_participation_rate

                        target_shares = (total_equity * position_size) / entry_price
                        if target_shares > max_shares_by_liquidity:
                            # Cap the position size percentage based on the liquidity limit
                            original_size_pct = position_size * 100
                            position_size = (max_shares_by_liquidity * entry_price) / total_equity
                            liquidity_adjustment_reason = f"capped at {max_shares_by_liquidity:.0f} shares (from {target_shares:.0f}) due to liquidity. Size reduced from {original_size_pct:.2f}% to {position_size*100:.2f}%."
                            logging.info(f"LIQUIDITY CAP for {symbol}: {liquidity_adjustment_reason}")

            except Exception as e:
                logging.warning(f"Could not apply liquidity constraint for {symbol}: {e}")
                liquidity_adjustment_reason = "error_in_analysis"

            # Use the robust, stateless core function for final share calculation
            shares, actual_position_value = core_calculate_position_size(
                symbol=symbol,
                entry_price=entry_price,
                position_size=position_size,
                total_equity=total_equity,
                buying_power=buying_power,
                force_minimum=force_minimum
            )
            result = (shares, {
                "position_size_pct": position_size * 100,
                "kelly_fraction": kelly_fraction,
                "position_value": actual_position_value,
                "volume_adjustment": volume_adjustment,
                "liquidity_adjustment": liquidity_adjustment_reason,
                "win_rate": float(win_rate),
                "avg_win": float(avg_win),
                "avg_loss": float(avg_loss),
                "symbol_win_rate": float(symbol_win_rate) if 'symbol_win_rate' in locals() else None,
                "volume_quality": volume_analysis.get('volume_quality_score', 50) if 'volume_analysis' in locals() else None,
                "volume_pattern": volume_analysis.get('volume_pattern', 'normal') if 'volume_analysis' in locals() else None,
                "sector": sector,
                "current_sector_exposure_pct": float(current_sector_pct),
                "buying_power": float(buying_power),
                "total_equity": float(total_equity)
            })
            self._log_tool_usage("calculate_position_size", args, result, time.time() - start_time)
            return result
        except Exception as e:
            logging.error(f"Position sizing error: {e}")
            result = (0, {"error": str(e)})
            self._log_tool_usage("calculate_position_size", args, result, time.time() - start_time)
            return result

    async def get_economic_calendar(self) -> List[Dict]:
        """
        Get economic calendar events.
        
        Returns:
            List of high-impact economic events
            
        Example Tool Input:
            {
                "tool_name": "get_economic_calendar",
                "args": {}
            }
            
        Example Output:
            [
                {
                    "date": "2024-01-16T13:30:00Z",
                    "event": "CPI m/m",
                    "impact": "high",
                    "country": "US"
                },
                {
                    "date": "2024-01-17T14:00:00Z",
                    "event": "Fed Chair Powell Speech",
                    "impact": "high",
                    "country": "US"
                }
            ]
        """
        import time
        start_time = time.time()
        args = {}
        try:
            await self._rate_limit('finnhub')
            today = dt.date.today()
            end_date = today + dt.timedelta(days=7)
            r = requests.get('https://finnhub.io/api/v1/calendar/economic',
                                params={'from': today.isoformat(), 'to': end_date.isoformat(), 'token': self.clients["finnhub_key"]},
                                timeout=10)
            if r.status_code == 200:
                events = r.json().get('economicCalendar', [])
                # Filter for high-impact US events
                high_impact_events = [
                    {"date": e['time'], "event": e['event'], "impact": e['impact'], "country": e['country']}
                    for e in events if e.get('impact') == 'high' and e.get('country') == 'US'
                ]
                result = high_impact_events
            else:
                result = [{"error": f"API Error, status {r.status_code}"}]
            self._log_tool_usage("get_economic_calendar", args, result, time.time() - start_time)
            return result
        except Exception as e:
            logging.error(f"Error fetching economic calendar: {e}")
            result = [{"error": str(e)}]
            self._log_tool_usage("get_economic_calendar", args, result, time.time() - start_time)
            return result

    async def analyze_volume_trend(self, symbol: str) -> Dict:
        """
        Deep volume trend analysis for comprehensive market understanding.
        
        Args:
            symbol: Stock symbol to analyze
            
        Returns:
            Volume trend analysis results
            
        Example Tool Input:
            {
                "tool_name": "analyze_volume_trend",
                "args": {"symbol": "AAPL"}
            }
            
        Example Output:
            {
                "symbol": "TSLA",
                "current_volume": 125000000,
                "volume_pattern": "strong_increase",
                "volume_ratio_20d": 2.15,
                "volume_ratio_50d": 1.45,
                "volume_ratio_100d": 1.25,
                "volume_on_up_days": 135000000,
                "volume_on_down_days": 115000000,
                "volume_momentum_5d": 25.5,
                "volume_momentum_10d": 15.2,
                "vpt_trend": "rising",
                "obv_trend": "rising",
                "avg_volume_20d": 58139500,
                "avg_volume_50d": 86206900,
                "volume_quality_score": 85,
                "smart_money_activity": "high"
            }
        """
        import time
        start_time = time.time()
        args = {"symbol": symbol}
        try:
            # Fetch extended data for comprehensive volume analysis
            loop = asyncio.get_event_loop()
            df = await loop.run_in_executor(
                None,
                lambda: yf.download(
                    tickers=symbol,
                    period="1y",  # Need longer period for volume patterns
                    interval="1d",
                    progress=False,
                    auto_adjust=False
                )
            )
            
            if df.empty or len(df) < 100:
                result = {"error": f"Insufficient data for volume analysis: {len(df) if not df.empty else 0} bars"}
                self._log_tool_usage("analyze_volume_trend", args, result, time.time() - start_time)
                return result
            
            # Handle MultiIndex columns
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = ['_'.join([str(i) for i in col]).strip('_') for col in df.columns.values]
                close_col = f'Close_{symbol}'
                volume_col = f'Volume_{symbol}'
                high_col = f'High_{symbol}'
                low_col = f'Low_{symbol}'
            else:
                close_col = 'Close'
                volume_col = 'Volume'
                high_col = 'High'
                low_col = 'Low'
            
            # Calculate multiple volume moving averages
            df['Volume_SMA_10'] = df[volume_col].rolling(10).mean()
            df['Volume_SMA_20'] = df[volume_col].rolling(20).mean()
            df['Volume_SMA_50'] = df[volume_col].rolling(50).mean()
            df['Volume_SMA_100'] = df[volume_col].rolling(100).mean()
            
            # Price change classification
            df['Price_Change'] = df[close_col].pct_change()
            df['Price_Direction'] = df['Price_Change'].apply(lambda x: 'up' if x > 0.005 else ('down' if x < -0.005 else 'flat'))
            
            # Volume-Price Trend (VPT) indicator
            df['VPT'] = (df['Price_Change'] * df[volume_col]).cumsum()
            
            # On-Balance Volume (OBV)
            df['OBV'] = self._calculate_obv(df[close_col], df[volume_col])
            
            # Volume analysis metrics
            current_volume = df[volume_col].iloc[-1]
            vol_sma_20 = df['Volume_SMA_20'].iloc[-1]
            vol_sma_50 = df['Volume_SMA_50'].iloc[-1]
            vol_sma_100 = df['Volume_SMA_100'].iloc[-1]
            
            # Volume ratios for trend strength
            volume_ratio_20 = current_volume / vol_sma_20 if vol_sma_20 > 0 else 1.0
            volume_ratio_50 = current_volume / vol_sma_50 if vol_sma_50 > 0 else 1.0
            volume_ratio_100 = current_volume / vol_sma_100 if vol_sma_100 > 0 else 1.0
            
            # Volume distribution analysis
            last_30_days = df.tail(30)
            volume_on_up_days = last_30_days[last_30_days['Price_Direction'] == 'up'][volume_col].mean()
            volume_on_down_days = last_30_days[last_30_days['Price_Direction'] == 'down'][volume_col].mean()
            volume_on_flat_days = last_30_days[last_30_days['Price_Direction'] == 'flat'][volume_col].mean()
            
            # Volume momentum (rate of change in volume)
            volume_roc_5 = ((current_volume / df[volume_col].iloc[-6]) - 1) * 100 if len(df) >= 6 else 0
            volume_roc_10 = ((vol_sma_20 / df['Volume_SMA_20'].iloc[-11]) - 1) * 100 if len(df) >= 31 else 0
            
            # VPT and OBV trend analysis
            vpt_current = df['VPT'].iloc[-1]
            vpt_20_ago = df['VPT'].iloc[-21] if len(df) >= 21 else vpt_current
            vpt_trend = 'rising' if vpt_current > vpt_20_ago else 'falling'
            
            obv_current = df['OBV'].iloc[-1]
            obv_20_ago = df['OBV'].iloc[-21] if len(df) >= 21 else obv_current
            obv_trend = 'rising' if obv_current > obv_20_ago else 'falling'
            
            # Volume pattern classification using config thresholds
            if volume_ratio_20 > CONFIG.get('volume_surge_ratio_20', 2.0) and volume_ratio_50 > CONFIG.get('volume_surge_ratio_50', 1.5):
                volume_pattern = 'explosive_surge'
            elif volume_ratio_20 > CONFIG.get('volume_surge_ratio_20', 2.0) and volume_ratio_50 > CONFIG.get('volume_increasing_ratio_50', 1.2):
                volume_pattern = 'strong_increase'
            elif volume_ratio_20 > CONFIG.get('volume_increasing_ratio_20', 1.2):
                volume_pattern = 'moderate_increase'
            elif volume_ratio_20 < CONFIG.get('volume_drying_up_ratio_20', 0.5):
                volume_pattern = 'severely_dry'
            elif volume_ratio_20 < CONFIG.get('volume_declining_ratio_20', 0.7):
                volume_pattern = 'declining'
            else:
                volume_pattern = 'normal'
            
            # Smart money vs retail money analysis
            high_volume_days = df[df[volume_col] > vol_sma_50 * 1.5]
            if len(high_volume_days) > 0:
                avg_price_move_high_vol = abs(high_volume_days['Price_Change']).mean() * 100
            else:
                avg_price_move_high_vol = 0
            
            low_volume_days = df[df[volume_col] < vol_sma_50 * 0.7]
            if len(low_volume_days) > 0:
                avg_price_move_low_vol = abs(low_volume_days['Price_Change']).mean() * 100
            else:
                avg_price_move_low_vol = 0
            
            # Volume quality score (0-100)
            quality_score = 0
            if volume_on_up_days > volume_on_down_days * 1.2:
                quality_score += 25
            if vpt_trend == 'rising' and obv_trend == 'rising':
                quality_score += 25
            if volume_pattern in ['strong_increase', 'explosive_surge']:
                quality_score += 20
            if avg_price_move_high_vol > avg_price_move_low_vol * 1.3:
                quality_score += 15
            if volume_ratio_20 > 1.0 and volume_ratio_50 > 1.0:
                quality_score += 15
            
            result = {
                "symbol": symbol,
                "current_volume": int(current_volume),
                "volume_pattern": volume_pattern,
                "volume_ratio_20d": float(volume_ratio_20),
                "volume_ratio_50d": float(volume_ratio_50),
                "volume_ratio_100d": float(volume_ratio_100),
                "volume_on_up_days": int(volume_on_up_days) if not pd.isna(volume_on_up_days) else 0,
                "volume_on_down_days": int(volume_on_down_days) if not pd.isna(volume_on_down_days) else 0,
                "volume_momentum_5d": float(volume_roc_5),
                "volume_momentum_10d": float(volume_roc_10),
                "vpt_trend": vpt_trend,
                "obv_trend": obv_trend,
                "avg_volume_20d": int(vol_sma_20),
                "avg_volume_50d": int(vol_sma_50),
                "volume_quality_score": int(quality_score),
                "smart_money_activity": "high" if avg_price_move_high_vol > avg_price_move_low_vol * 1.5 else "normal",
                "volume_on_flat_days": int(volume_on_flat_days) if not pd.isna(volume_on_flat_days) else 0,
            }
            
            self._log_tool_usage("analyze_volume_trend", args, result, time.time() - start_time)
            return result
            
        except Exception as e:
            logging.error(f"Error in volume trend analysis for {symbol}: {e}")
            result = {"error": str(e)}
            self._log_tool_usage("analyze_volume_trend", args, result, time.time() - start_time)
            return result

    async def calculate_risk_metrics(self) -> Dict:
        """
        Calculate portfolio risk metrics including beta, drawdown, Sharpe, VaR, CVaR, and correlation matrix.
        """
        import time
        start_time = time.time()
        args = {}
        try:
            loop = asyncio.get_running_loop()
            
            # Add timeout for the entire operation
            async with asyncio.timeout(60):  # 60 second timeout
                account = await loop.run_in_executor(None, self.clients["alpaca"].get_account)
                history_request = GetPortfolioHistoryRequest(
                    period="1M",  # Reduced from 3M to 1M for faster processing
                    timeframe="1D",
                )
                portfolio_history = await loop.run_in_executor(
                    None,
                    lambda: self.clients["alpaca"].get_portfolio_history(history_request),
                )
                equity = getattr(portfolio_history, "equity", None)
                timestamps = getattr(portfolio_history, "timestamp", None)
                if not equity or len(equity) < 2:
                    result = {
                        "portfolio_beta": 1.0,
                        "max_drawdown": 0.0,
                        "current_drawdown": 0.0,
                        "sharpe_ratio": 0.0,
                        "sortino_ratio": 0.0,
                        "var_95": 0.0,
                        "cvar_95": 0.0,
                        "correlation_matrix": None,
                        "twrr_pct": 0.0,
                        "account_equity": float(account.equity),
                        "buying_power": float(account.buying_power),
                    }
                    self._log_tool_usage("calculate_risk_metrics", args, result, time.time() - start_time)
                    return result

                equity_series = pd.Series([float(x) for x in equity])
                if timestamps and len(timestamps) == len(equity):
                    equity_series.index = pd.to_datetime(timestamps, unit="s", utc=True)
                    equity_series = equity_series.sort_index()
                returns = equity_series.pct_change().dropna()
                
                if returns.empty:
                    result = {
                        "portfolio_beta": 1.0,
                        "max_drawdown": 0.0,
                        "current_drawdown": 0.0,
                        "sharpe_ratio": 0.0,
                        "sortino_ratio": 0.0,
                        "var_95": 0.0,
                        "cvar_95": 0.0,
                        "correlation_matrix": None,
                        "twrr_pct": 0.0,
                        "account_equity": float(account.equity),
                        "buying_power": float(account.buying_power),
                    }
                    self._log_tool_usage("calculate_risk_metrics", args, result, time.time() - start_time)
                    return result

                # Calculate basic metrics using equity curve directly
                rolling_max = equity_series.cummax()
                drawdown = (equity_series / rolling_max) - 1
                max_drawdown = abs(drawdown.min()) * 100
                current_drawdown = abs(drawdown.iloc[-1]) * 100

                # Sharpe and Sortino ratios
                if returns.std() > 0:
                    sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)
                else:
                    sharpe_ratio = 0.0

                downside_returns = returns[returns < 0]
                if len(downside_returns) > 0 and downside_returns.std() > 0:
                    sortino_ratio = (returns.mean() / downside_returns.std()) * np.sqrt(252)
                else:
                    sortino_ratio = sharpe_ratio

                # === VaR and CVaR Calculation ===
                returns_pct = returns * 100
                var_95 = np.percentile(returns_pct, 5) if not returns_pct.empty else 0.0
                # CVaR is the average of the returns that are worse than the VaR
                cvar_95 = returns_pct[returns_pct < var_95].mean() if not returns_pct[returns_pct < var_95].empty else var_95

                # Portfolio beta calculation (simplified for speed)
                portfolio_beta = 1.0  # Default value to avoid timeout
                try:
                    async with asyncio.timeout(20):  # 20 second timeout for beta
                        spy_df = yf.download(
                            tickers="SPY",
                            period="50d",  # Reduced from 100d
                            interval="1d",
                            progress=False,
                            auto_adjust=False
                        )
                        if not spy_df.empty:
                            if isinstance(spy_df.columns, pd.MultiIndex):
                                spy_df.columns = ['_'.join([str(i) for i in col]).strip('_') for col in spy_df.columns.values]
                                close_col = 'Close_SPY'
                            else:
                                close_col = 'Close'
                            spy_returns = spy_df[close_col].pct_change().dropna()

                            portfolio_returns = returns.copy()
                            if isinstance(portfolio_returns.index, pd.DatetimeIndex):
                                portfolio_returns.index = portfolio_returns.index.tz_localize(None).normalize()
                            if isinstance(spy_returns.index, pd.DatetimeIndex):
                                spy_returns.index = spy_returns.index.tz_localize(None).normalize()

                            aligned = pd.concat([portfolio_returns, spy_returns], axis=1, join="inner").dropna()
                            if len(aligned) >= 20:
                                covariance = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1], ddof=1)[0, 1]
                                spy_variance = np.var(aligned.iloc[:, 1], ddof=1)
                                portfolio_beta = covariance / spy_variance if spy_variance > 0 else 1.0
                except asyncio.TimeoutError:
                    logging.warning("Beta calculation timed out, using default value")
                except Exception as e:
                    logging.warning(f"Beta calculation error: {e}")

                # === Portfolio Correlation Matrix Calculation (simplified) ===
                correlation_matrix_dict = None
                try:
                    async with asyncio.timeout(30):  # 30 second timeout for correlation
                        positions = await loop.run_in_executor(None, self._alpaca_list_positions)
                        position_symbols = [p.symbol for p in positions]

                        if len(position_symbols) > 1 and len(position_symbols) <= 10:  # Limit to 10 positions max
                            pos_data = await loop.run_in_executor(
                                None,
                                lambda: yf.download(
                                    tickers=position_symbols,
                                    period="50d",  # Reduced from 100d
                                    interval="1d",
                                    progress=False,
                                    auto_adjust=True
                                )
                            )
                            # Use daily percentage change for returns
                            returns_df = pos_data['Close'].pct_change().dropna()
                            correlation_matrix = returns_df.corr()
                            correlation_matrix_dict = correlation_matrix.to_dict()
                except asyncio.TimeoutError:
                    logging.warning("Correlation matrix calculation timed out")
                except Exception as e:
                    logging.warning(f"Could not calculate portfolio correlation matrix: {e}")

                # Ensure no infinity or NaN values
                max_drawdown = max_drawdown if np.isfinite(max_drawdown) else 0.0
                current_drawdown = current_drawdown if np.isfinite(current_drawdown) else 0.0
                sharpe_ratio = sharpe_ratio if np.isfinite(sharpe_ratio) else 0.0
                sortino_ratio = sortino_ratio if np.isfinite(sortino_ratio) else 0.0
                var_95 = var_95 if np.isfinite(var_95) else 0.0
                cvar_95 = cvar_95 if np.isfinite(cvar_95) else var_95

                # === TWRR Calculation ===
                twrr = 0.0
                cashflow = getattr(portfolio_history, "cashflow", None)
                if cashflow:
                    net_cashflows = [0.0] * len(equity_series)
                    if isinstance(cashflow, dict):
                        for flows in cashflow.values():
                            if not flows:
                                continue
                            for i, cf in enumerate(flows):
                                if i < len(net_cashflows) and cf is not None:
                                    net_cashflows[i] += float(cf)
                    hprs = []
                    equity_values = equity_series.values
                    for i in range(1, len(equity_values)):
                        start_val = equity_values[i - 1]
                        if start_val <= 0:
                            continue
                        end_val = equity_values[i]
                        cf = net_cashflows[i] if i < len(net_cashflows) else 0.0
                        hprs.append((end_val - cf) / start_val)
                    if hprs:
                        twrr = (np.prod(hprs) - 1) * 100
                elif self.memory:
                    daily_snapshots = self.memory.memory.get("daily_snapshots", [])
                    if len(daily_snapshots) > 1:
                        twrr = calculate_twrr(daily_snapshots)

                result = {
                    "portfolio_beta": portfolio_beta,
                    "max_drawdown": max_drawdown,
                    "current_drawdown": current_drawdown,
                    "sharpe_ratio": sharpe_ratio,
                    "sortino_ratio": sortino_ratio,
                    "var_95": var_95,
                    "cvar_95": cvar_95,
                    "correlation_matrix": correlation_matrix_dict,
                    "twrr_pct": twrr,
                    "account_equity": float(account.equity),
                    "buying_power": float(account.buying_power),
                }
                self._log_tool_usage("calculate_risk_metrics", args, result, time.time() - start_time)
                return result

        except asyncio.TimeoutError:
            logging.error("Risk metrics calculation timed out")
            result = {
                "portfolio_beta": 1.0,
                "max_drawdown": 0.0,
                "current_drawdown": 0.0,
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "var_95": 0.0,
                "cvar_95": 0.0,
                "correlation_matrix": None,
                "twrr_pct": 0.0,
                "error": "Calculation timed out",
            }
            self._log_tool_usage("calculate_risk_metrics", args, result, time.time() - start_time)
            return result
        except Exception as e:
            logging.error(f"Risk metrics error: {e}")
            result = {
                "portfolio_beta": 1.0,
                "max_drawdown": 0.0,
                "current_drawdown": 0.0,
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "var_95": 0.0,
                "cvar_95": 0.0,
                "correlation_matrix": None,
                "twrr_pct": 0.0,
                "error": str(e),
            }
            self._log_tool_usage("calculate_risk_metrics", args, result, time.time() - start_time)
            return result

    async def analyze_sector_exposure(self) -> Dict:
        """
        Analyze current sector allocation.
        
        Returns:
            Sector exposure analysis with percentages and concentration warnings
        
        Example Tool Input:
            {
                "tool_name": "analyze_sector_exposure",
                "args": {}
            }
        
        Example Output:
            {
                "sectors": {
                    "Technology": 35.2,
                    "Healthcare": 22.1,
                    "Financial": 18.5,
                    "Consumer Discretionary": 12.3
                },
                "cash_percentage": 11.9,
                "total_equity": 125000.0,
                "position_count": 8,
                "concentrated_sectors": ["Technology"]
            }
        """
        import time
        start_time = time.time()
        args = {}
        try:
            loop = asyncio.get_running_loop()
            
            # Add timeout for the entire operation
            async with asyncio.timeout(20):  # 20 second timeout
                positions = await loop.run_in_executor(None, self._alpaca_list_positions)
                account = await loop.run_in_executor(None, self.clients["alpaca"].get_account)
                total_equity = float(account.equity)  # Total portfolio value including cash
                cash_value = float(account.cash)
                logging.info(f"[DEBUG] analyze_sector_exposure: cash_value={cash_value}, total_equity={total_equity}")
                sector_exposure = defaultdict(float)

                for pos in positions:
                    market_value = float(pos.market_value)
                    sector = SECTOR_MAP.get(pos.symbol, "Other")
                    sector_exposure[sector] += market_value
                
                # Convert to percentages of total portfolio (including cash)
                sector_pct = {}
                for sector, value in sector_exposure.items():
                    sector_pct[sector] = (value / total_equity * 100) if total_equity > 0 else 0.0
                
                # Calculate cash percentage
                cash_pct = (cash_value / total_equity * 100) if total_equity > 0 else 0.0
                
                result = {
                    "sectors": dict(sector_pct),
                    "cash_percentage": cash_pct,
                    "total_equity": total_equity,
                    "position_count": len(positions),
                    "concentrated_sectors": [s for s, pct in sector_pct.items() if pct > CONFIG.get('max_sector_exposure_pct', 25.0)]
                }
                self._log_tool_usage("analyze_sector_exposure", args, result, time.time() - start_time)
                return result
                
        except asyncio.TimeoutError:
            logging.error("Sector exposure analysis timed out")
            result = {"sectors": {}, "cash_percentage": 0.0, "total_equity": 0.0, "position_count": 0, "concentrated_sectors": []}
            self._log_tool_usage("analyze_sector_exposure", args, result, time.time() - start_time)
            return result
        except Exception as e:
            logging.error(f"Sector analysis error: {e}")
            result = {"sectors": {}, "cash_percentage": 0.0, "total_equity": 0.0, "position_count": 0, "concentrated_sectors": []}
            self._log_tool_usage("analyze_sector_exposure", args, result, time.time() - start_time)
            return result

    async def get_portfolio_state(self) -> List[Dict]:
        """
        Enhanced portfolio state with technical indicators.
        
        Returns:
            List of portfolio positions with technical analysis
        
        Example Tool Input:
            {
                "tool_name": "get_portfolio_state",
                "args": {}
            }
        
        Example Output:
            [
                {
                    "symbol": "NVDA",
                    "qty": 25.0,
                    "avg_entry_price": 485.50,
                    "current_price": 495.75,
                    "market_value": 12393.75,
                    "unrealized_pl": 256.25,
                    "unrealized_pl_pct": 2.1,
                    "position_pct": 9.9,
                    "sector": "Technology",
                    "technical_signals": {...},
                    "take_profit_order_id": "12345",
                    "stop_loss_order_id": "12346"
                }
            ]
        """
        import time
        start_time = time.time()
        args = {}
        try:
            loop = asyncio.get_running_loop()
            
            # Add timeout for the entire operation
            async with asyncio.timeout(40):  # 40 second timeout
                start_time = time.time()
            args = {}
            try:
                loop = asyncio.get_running_loop()
                
                # Add timeout for the entire operation
                async with asyncio.timeout(40):  # 40 second timeout
                    positions = await loop.run_in_executor(None, self._alpaca_list_positions)
                    orders = await loop.run_in_executor(None, lambda: self._alpaca_list_orders(QueryOrderStatus.OPEN, True))
                    account = await loop.run_in_executor(None, self.clients["alpaca"].get_account)
                    total_equity = float(account.equity)
                    
                    state = []
                    expanded_orders = []
                    for order in orders:
                        expanded_orders.append(order)
                        legs = getattr(order, "legs", None)
                        if legs:
                            expanded_orders.extend(legs)

                    def _order_side_value(order_obj):
                        side = getattr(order_obj, "side", None)
                        if hasattr(side, "value"):
                            return side.value
                        return str(side).lower() if side is not None else ""

                    def _order_type_value(order_obj):
                        otype = getattr(order_obj, "type", None)
                        if hasattr(otype, "value"):
                            return otype.value
                        return str(otype).lower() if otype is not None else ""

                    for pos in positions:
                        position_reason = ""
                        if self.memory:
                            last_trade = self.memory.get_last_trade_for_symbol(pos.symbol)
                            if last_trade and last_trade.get("reason"):
                                position_reason = last_trade.get("reason", "")
                            else:
                                for item in self.memory.get_watchlist():
                                    if str(item.get("symbol", "")).upper() == str(pos.symbol).upper():
                                        position_reason = item.get("reason", "")
                                        break

                        # Get technical signals for position (with timeout)
                        try:
                            async with asyncio.timeout(10):  # 10 second timeout per position
                                tech_signals = await self.analyze_technicals(pos.symbol)
                        except asyncio.TimeoutError:
                            tech_signals = {"error": "Technical analysis timed out"}
                        except Exception as e:
                            tech_signals = {"error": f"Technical analysis failed: {str(e)}"}
                        
                        # Find related orders
                        related_orders = [o for o in expanded_orders if getattr(o, "symbol", None) == pos.symbol]
                        take_profit_order_id = next(
                            (
                                str(o.id)
                                for o in related_orders
                                if _order_side_value(o) == OrderSide.SELL.value
                                and _order_type_value(o) == OrderType.LIMIT.value
                            ),
                            None,
                        )
                        stop_loss_order_id = next(
                            (
                                str(o.id)
                                for o in related_orders
                                if _order_side_value(o) == OrderSide.SELL.value
                                and _order_type_value(o)
                                in {OrderType.STOP.value, OrderType.TRAILING_STOP.value, OrderType.STOP_LIMIT.value}
                            ),
                            None,
                        )
                        
                        # Calculate position metrics
                        position_pct = (float(pos.market_value) / total_equity * 100) if total_equity > 0 else 0
                        
                        state.append({
                            "symbol": pos.symbol,
                            "qty": float(pos.qty),
                            "avg_entry_price": float(pos.avg_entry_price),
                            "current_price": float(pos.current_price),
                            "market_value": float(pos.market_value),
                            "unrealized_pl": float(pos.unrealized_pl),
                            "unrealized_pl_pct": float(pos.unrealized_plpc),
                            "position_pct": position_pct,
                            "sector": SECTOR_MAP.get(pos.symbol, "Other"),
                            "position_reason": position_reason,
                            "technical_signals": tech_signals,
                            "take_profit_order_id": take_profit_order_id,
                            "stop_loss_order_id": stop_loss_order_id,
                        })
                    
                    self._log_tool_usage("get_portfolio_state", args, state, time.time() - start_time)
                    return state
                
            except asyncio.TimeoutError:
                logging.error("Portfolio state calculation timed out")
                result = []
                self._log_tool_usage("get_portfolio_state", args, result, time.time() - start_time)
                return result
        except Exception as e:
            logging.error(f"Portfolio state error: {e}")
            result = []
            self._log_tool_usage("get_portfolio_state", args, result, time.time() - start_time)
            return result

    async def analyze_intermarket_correlation(self) -> Dict:
        """
        Analyze correlations between different asset classes (equities, bonds, commodities, currencies).
        Returns:
            Intermarket correlation analysis and interpretation
        Example Tool Input:
            {
                "tool_name": "analyze_intermarket_correlation",
                "args": {}
            }
        Example Output:
            {
                "correlations": {
                    "equities_bonds": -0.25,
                    "equities_gold": -0.15,
                    "equities_dollar": -0.18
                },
                "interpretation": "Risk-off environment (equities and bonds negatively correlated)",
                "risk_off_environment": true,
                "dollar_strength": true
            }
        """
        import time
        import asyncio
        start_time = time.time()
        args = {}
        try:
            symbols = {
                'SPY': 'Equities',
                'TLT': 'Bonds',
                'GLD': 'Gold',
                'UUP': 'Dollar',
                'VIX': 'Volatility'
            }
            correlations = {}
            price_data = {}
            missing_assets = []
            missing_errors = {}
            # Helper: retry logic for yfinance fetch, with fallback to shorter periods
            async def fetch_with_retries(symbol, asset_class, max_retries=3, delay=1.0):
                last_exception = None
                periods = ["100d", "30d", "10d"]
                for period in periods:
                    for attempt in range(max_retries):
                        try:
                            df = yf.download(symbol, period=period, interval="1d", progress=False, auto_adjust=False)
                            if not df.empty:
                                if isinstance(df.columns, pd.MultiIndex):
                                    df.columns = ['_'.join([str(i) for i in col]).strip('_') for col in df.columns.values]
                                    close_col = f'Close_{symbol}'
                                else:
                                    close_col = 'Close'
                                return df[close_col].pct_change().dropna(), None
                            else:
                                last_exception = f"Empty DataFrame for {symbol} with period {period}"
                        except Exception as e:
                            last_exception = str(e)
                        if attempt < max_retries - 1:
                            await asyncio.sleep(delay)
                return None, last_exception
            for symbol, asset_class in symbols.items():
                if symbol == 'VIX':
                    continue
                returns, err = await fetch_with_retries(symbol, asset_class)
                if returns is not None and not returns.empty:
                    price_data[asset_class] = returns
                else:
                    missing_assets.append(asset_class)
                    missing_errors[asset_class] = err or 'Unknown error'
            if len(price_data) >= 2:
                all_returns = pd.DataFrame(price_data)
                correlation_matrix = all_returns.corr()
                if 'Equities' in correlation_matrix.index and 'Bonds' in correlation_matrix.columns:
                    correlations['equities_bonds'] = float(correlation_matrix.loc['Equities', 'Bonds'])
                if 'Equities' in correlation_matrix.index and 'Gold' in correlation_matrix.columns:
                    correlations['equities_gold'] = float(correlation_matrix.loc['Equities', 'Gold'])
                if 'Equities' in correlation_matrix.index and 'Dollar' in correlation_matrix.columns:
                    correlations['equities_dollar'] = float(correlation_matrix.loc['Equities', 'Dollar'])
                if 'Bonds' in correlation_matrix.index and 'Gold' in correlation_matrix.columns:
                    correlations['bonds_gold'] = float(correlation_matrix.loc['Bonds', 'Gold'])
                interpretation = self._interpret_correlations(correlations)
                result = {
                    "correlations": correlations,
                    "interpretation": interpretation,
                    "risk_off_environment": correlations.get('equities_bonds', 0) < -0.3,
                    "dollar_strength": correlations.get('equities_dollar', 0) < -0.2,
                    "correlation_matrix": correlation_matrix.to_dict() if 'correlation_matrix' in locals() else {},
                    "asset_performance": {
                        asset: {
                            "mean_return": float(returns.mean()),
                            "volatility": float(returns.std()),
                            "sharpe_ratio": float(returns.mean() / returns.std()) if returns.std() > 0 else 0.0
                        } for asset, returns in price_data.items()
                    },
                    "data_points": len(next(iter(price_data.values()))) if price_data else 0,
                    "missing_assets": missing_assets,
                }
                if missing_assets:
                    result["warning"] = f"Some assets missing: {missing_assets}. Errors: {missing_errors}"
                self._log_tool_usage("analyze_intermarket_correlation", args, result, time.time() - start_time)
                return result
            else:
                import logging
                logging.error(f"Insufficient data for correlation analysis. Missing: {missing_assets}. Errors: {missing_errors}")
                result = {"error": f"Insufficient data for correlation analysis. Missing: {missing_assets}. Errors: {missing_errors}"}
                self._log_tool_usage("analyze_intermarket_correlation", args, result, time.time() - start_time)
                return result
        except Exception as e:
            import logging
            logging.error(f"Intermarket correlation error: {e}")
            result = {"error": str(e)}
            self._log_tool_usage("analyze_intermarket_correlation", args, result, time.time() - start_time)
            return result

    def _interpret_correlations(self, correlations: Dict[str, float]) -> str:
        """Interpret correlation patterns for market regime analysis."""
        interpretations = []
        
        eq_bonds = correlations.get('equities_bonds', 0)
        if eq_bonds < -0.3:
            interpretations.append("Risk-off environment (equities and bonds negatively correlated)")
        elif eq_bonds > 0.3:
            interpretations.append("Risk-on environment (equities and bonds positively correlated)")
        
        eq_gold = correlations.get('equities_gold', 0)
        if eq_gold < -0.2:
            interpretations.append("Flight to safety (equities and gold negatively correlated)")
        
        eq_dollar = correlations.get('equities_dollar', 0)
        if eq_dollar < -0.2:
            interpretations.append("Dollar strength pressuring equities")
        elif eq_dollar > 0.2:
            interpretations.append("Dollar weakness supporting equities")
        
        if not interpretations:
            interpretations.append("Normal correlation patterns")
        
        return "; ".join(interpretations)

    async def calculate_portfolio_beta(self) -> Dict:
        """
        Calculate portfolio beta relative to SPY.
        
        Returns:
            Portfolio beta calculation and interpretation
        
        Example Tool Input:
            {
                "tool_name": "calculate_portfolio_beta",
                "args": {}
            }
        
        Example Output:
            {
                "portfolio_beta": 1.15,
                "interpretation": "Moderate beta portfolio - similar volatility to market",
                "market_relative_volatility": "normal"
            }
        """
        import time
        start_time = time.time()
        args = {}
        try:
            # Get portfolio returns
            loop = asyncio.get_running_loop()
            history_request = GetPortfolioHistoryRequest(period="1M", timeframe="1D")
            portfolio_history = await loop.run_in_executor(
                None,
                lambda: self.clients["alpaca"].get_portfolio_history(history_request),
            )

            equity = getattr(portfolio_history, "equity", None)
            timestamps = getattr(portfolio_history, "timestamp", None)
            if not equity or len(equity) < 2:
                result = {"portfolio_beta": 1.0, "interpretation": "Insufficient data"}
                self._log_tool_usage("calculate_portfolio_beta", args, result, time.time() - start_time)
                return result

            equity_series = pd.Series([float(x) for x in equity])
            if timestamps and len(timestamps) == len(equity):
                equity_series.index = pd.to_datetime(timestamps, unit="s", utc=True)
                equity_series = equity_series.sort_index()
            portfolio_returns = equity_series.pct_change().dropna()
            
            # Get SPY returns
            spy_df = yf.download(
                tickers="SPY",
                period="100d",
                interval="1d",
                progress=False,
                auto_adjust=False
            )
            
            if spy_df.empty:
                result = {"portfolio_beta": 1.0, "interpretation": "Could not fetch market data"}
                self._log_tool_usage("calculate_portfolio_beta", args, result, time.time() - start_time)
                return result
            
            if isinstance(spy_df.columns, pd.MultiIndex):
                spy_df.columns = ['_'.join([str(i) for i in col]).strip('_') for col in spy_df.columns.values]
                close_col = 'Close_SPY'
            else:
                close_col = 'Close'
            
            spy_returns = spy_df[close_col].pct_change().dropna()

            if isinstance(portfolio_returns.index, pd.DatetimeIndex):
                portfolio_returns.index = portfolio_returns.index.tz_localize(None).normalize()
            if isinstance(spy_returns.index, pd.DatetimeIndex):
                spy_returns.index = spy_returns.index.tz_localize(None).normalize()
            
            # Calculate beta
            aligned = pd.concat([portfolio_returns, spy_returns], axis=1, join="inner").dropna()
            if len(aligned) >= 20:
                covariance = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1], ddof=1)[0, 1]
                spy_variance = np.var(aligned.iloc[:, 1], ddof=1)

                if spy_variance > 0:
                    portfolio_beta = covariance / spy_variance
                else:
                    portfolio_beta = 1.0
            else:
                portfolio_beta = 1.0
            
            # Interpret beta
            if portfolio_beta > 1.2:
                interpretation = "High beta portfolio - more volatile than market"
            elif portfolio_beta > 0.8:
                interpretation = "Moderate beta portfolio - similar volatility to market"
            elif portfolio_beta > 0.5:
                interpretation = "Low beta portfolio - less volatile than market"
            else:
                interpretation = "Very low beta portfolio - defensive positioning"
            
            result = {
                "portfolio_beta": float(portfolio_beta),
                "interpretation": interpretation,
                "market_relative_volatility": "high" if portfolio_beta > 1.2 else ("low" if portfolio_beta < 0.8 else "normal")
            }
            
            self._log_tool_usage("calculate_portfolio_beta", args, result, time.time() - start_time)
            return result
            
        except Exception as e:
            logging.error(f"Portfolio beta calculation error: {e}")
            result = {"portfolio_beta": 1.0, "interpretation": f"Error: {str(e)}"}
            self._log_tool_usage("calculate_portfolio_beta", args, result, time.time() - start_time)
            return result

    async def smart_order_entry(
        self,
        symbol: str,
        shares: float,
        order_type: str = "smart",
        take_profit_pct: float = None,
        stop_loss_pct: float = None,
        limit_price: float = None,
        reason: str = "",
    ) -> Dict:
        """
        Smart order entry with bracket orders and risk management.
        
        Args:
            symbol: Stock symbol
            shares: Number of shares to buy
            order_type: Order type (default: "smart")
            take_profit_pct: Take profit percentage (optional)
            stop_loss_pct: Stop loss percentage (optional)
            
        Returns:
            Order details and status
        
        Example Tool Input:
            {
                "tool_name": "smart_order_entry",
                "args": {
                    "symbol": "NVDA",
                    "shares": 25,
                    "order_type": "smart",
                    "take_profit_pct": 5.0,
                    "stop_loss_pct": 2.0
                }
            }
        
        Example Output:
            {
                "status": "success",
                "symbol": "NVDA",
                "qty": 25,
                "order_type": "bracket",
                "entry_order_id": "12345",
                "stop_price": 470.94,
                "take_profit_price": 509.78,
                "stop_order_id": "12346",
                "take_profit_order_id": "12347"
            }
        """
        import time
        start_time = time.time()
        args = {
            "symbol": symbol,
            "shares": shares,
            "order_type": order_type,
            "take_profit_pct": take_profit_pct,
            "stop_loss_pct": stop_loss_pct,
            "limit_price": limit_price,
            "reason": reason,
        }
        
        print(f"[TRADE] Starting smart_order_entry for {symbol}: {shares} shares")
        
        try:
            await self._rate_limit('alpaca')
            
            # Get current quote for price with error handling
            estimated_slippage = 0.0
            try:
                print(f"[TRADE] Getting quote for {symbol}")
                loop = asyncio.get_running_loop()
                data_client = self.clients.get("alpaca_data")
                if not data_client:
                    raise RuntimeError("Alpaca data client not configured")

                quote_dict = await loop.run_in_executor(
                    None,
                    lambda: data_client.get_stock_latest_quote(
                        StockLatestQuoteRequest(symbol_or_symbols=symbol)
                    ),
                )
                quote = quote_dict.get(symbol) or quote_dict.get(symbol.upper())
                if not quote:
                    raise RuntimeError(f"No quote data returned for {symbol}")

                current_price = quote.ask_price if quote.ask_price else quote.bid_price
                print(f"[TRADE] Got quote for {symbol}: ${current_price}")
                
                # === SLIPPAGE ESTIMATION ===
                if quote.ask_price is not None and quote.bid_price is not None:
                    spread = quote.ask_price - quote.bid_price
                    # Simple model: assume we capture half the spread as slippage
                    slippage_per_share = spread / 2
                    estimated_slippage = slippage_per_share * shares
                    logging.info(f"Estimated slippage for {symbol}: ${estimated_slippage:.2f} total (${slippage_per_share:.4f}/share)")
                    
            except Exception as quote_error:
                print(f"[TRADE] Quote error for {symbol}: {quote_error}")
                data_client = self.clients.get("alpaca_data")
                if not data_client:
                    raise
                bars = await loop.run_in_executor(
                    None,
                    lambda: data_client.get_stock_latest_bar(
                        StockLatestBarRequest(symbol_or_symbols=symbol)
                    ),
                )
                bar = bars.get(symbol) or bars.get(symbol.upper())
                if not bar:
                    raise RuntimeError(f"No bar data returned for {symbol}")
                current_price = bar.close
                print(f"[TRADE] Using bar data for {symbol}: ${current_price}")
            
            tech = await self.analyze_technicals(symbol)
            atr_pct = tech.get('volatility', 0.02)
            
            # Allow override of stop_loss_pct and take_profit_pct
            stop_loss_pct = stop_loss_pct if stop_loss_pct is not None else atr_pct * 1.5
            take_profit_pct = take_profit_pct if take_profit_pct is not None else atr_pct * 2.5
            
            # Ensure percentages are positive and reasonable
            stop_loss_pct = max(0.01, min(stop_loss_pct, 0.10))  # Between 1% and 10%
            take_profit_pct = max(0.01, min(take_profit_pct, 0.20))  # Between 1% and 20%
            
            pricing_base = limit_price if limit_price else current_price
            stop_price = round(pricing_base * (1 - stop_loss_pct), 2)
            take_profit_price = round(pricing_base * (1 + take_profit_pct), 2) if take_profit_pct else None

            print(f"[TRADE] Calculated prices for {symbol}: entry=${pricing_base}, stop=${stop_price}, take_profit=${take_profit_price}")
            
            stop_loss_req = StopLossRequest(stop_price=stop_price)
            take_profit_req = TakeProfitRequest(limit_price=take_profit_price) if take_profit_price else None

            if order_type == "limit" or limit_price is not None:
                if not limit_price:
                    raise ValueError("limit_price is required for limit orders")
                order_request = LimitOrderRequest(
                    symbol=symbol,
                    qty=shares,
                    side=OrderSide.BUY,
                    type=OrderType.LIMIT,
                    time_in_force=TimeInForce.GTC,
                    order_class=OrderClass.BRACKET,
                    limit_price=limit_price,
                    stop_loss=stop_loss_req,
                    take_profit=take_profit_req,
                )
            else:
                order_request = MarketOrderRequest(
                    symbol=symbol,
                    qty=shares,
                    side=OrderSide.BUY,
                    type=OrderType.MARKET,
                    time_in_force=TimeInForce.GTC,
                    order_class=OrderClass.BRACKET,
                    stop_loss=stop_loss_req,
                    take_profit=take_profit_req,
                )
            
            print(f"[TRADE] Submitting bracket order for {symbol}: {order_request}")
            
            loop = asyncio.get_running_loop()
            order = await loop.run_in_executor(None, lambda: self.clients["alpaca"].submit_order(order_request))
            
            order_id = str(order.id) if getattr(order, "id", None) is not None else None
            print(f"[TRADE] Order submitted successfully for {symbol}: order_id={order_id}")
            
            result = {
                "status": "success",
                "message": "Smart order placed successfully",
                "order_id": order_id,
                "symbol": symbol,
                "shares": shares,
                "estimated_fill_price": pricing_base,
                "order_type": order_type,
                "take_profit_price": take_profit_price,
                "stop_loss_price": stop_price,
                "estimated_cost": shares * pricing_base,
                "estimated_slippage": estimated_slippage,
                "timestamp": dt.datetime.now(dt.timezone.utc).isoformat()
            }
            if hasattr(order, 'legs') and order.legs:
                for leg in order.legs:
                    if getattr(leg, 'type', None) == 'stop':
                        result['stop_order_id'] = str(leg.id) if getattr(leg, "id", None) is not None else None
                    if getattr(leg, 'type', None) == 'limit':
                        result['take_profit_order_id'] = str(leg.id) if getattr(leg, "id", None) is not None else None
            
            print(f"[TRADE] Final result for {symbol}: {result}")
            
            # Verify position was created (poll broker directly; avoid heavy portfolio call)
            try:
                result["position_verified"] = False
                order_status = None
                if order_id:
                    try:
                        order_info = await loop.run_in_executor(
                            None,
                            lambda: self._alpaca_get_order(order_id),
                        )
                        order_status = getattr(order_info, "status", None)
                        result["order_status"] = order_status
                        result["filled_qty"] = getattr(order_info, "filled_qty", None)
                    except Exception as order_status_error:
                        result["order_status_error"] = str(order_status_error)

                for attempt in range(5):
                    await asyncio.sleep(1 + attempt)  # progressive backoff
                    try:
                        position = await loop.run_in_executor(
                            None,
                            lambda: self._alpaca_get_position(symbol),
                        )
                        if position:
                            result["position_verified"] = True
                            result["actual_shares"] = float(position.qty)
                            result["actual_entry_price"] = float(position.avg_entry_price)
                            print(f"[TRADE] Position verified: {position.qty} shares at ${position.avg_entry_price}")

                            # Log the trade with estimated slippage
                            if self.memory:
                trade_log_data = {
                    "symbol": symbol,
                    "qty": result["actual_shares"],
                    "price": result["actual_entry_price"],
                    "side": "buy",
                    "order_id": order_id,
                    "reason": reason or "Agent execution",
                    "estimated_slippage": estimated_slippage,
                }
                                self.memory.log_trade(trade_log_data)
                            break
                    except Exception:
                        continue

                if not result["position_verified"]:
                    if order_status and str(order_status).lower() not in {"filled", "partially_filled"}:
                        result["warning"] = "Order accepted but not filled yet; position may appear shortly"
                    else:
                        result["warning"] = "Position not found after verification attempts"
                    print(f"[TRADE] WARNING: Position not verified for {symbol}")
            except Exception as verify_error:
                result["position_verified"] = False
                result["verification_error"] = str(verify_error)
                print(f"[TRADE] Position verification failed: {verify_error}")
            
            self._log_tool_usage("smart_order_entry", args, result, time.time() - start_time)
            return result
            
        except Exception as e:
            print(f"[TRADE] ERROR in smart_order_entry for {symbol}: {e}")
            logging.error(f"Bracket order entry error: {e}")
            result = {"status": "error", "message": str(e)}
            self._log_tool_usage("smart_order_entry", args, result, time.time() - start_time)
            return result

    async def buy_option(self, symbol: str, strike: float, expiration_date: str, quantity: int, contract_type: str, reason: str = "") -> Dict:
        """
        Buy an option contract.
        
        Args:
            symbol: The underlying stock symbol
            strike: The strike price of the option
            expiration_date: The expiration date of the option in YYYY-MM-DD format
            quantity: The number of option contracts to buy
            contract_type: The type of option to buy, either 'put' or 'call'
            
        Returns:
            A dictionary containing the order details.
        """
        start_time = time.time()
        args = {"symbol": symbol, "strike": strike, "expiration_date": expiration_date, "quantity": quantity, "contract_type": contract_type, "reason": reason}
        
        try:
            trading_client = self.clients.get("alpaca")
            if not trading_client:
                raise RuntimeError("Alpaca trading client not configured")
            
            strike_str = f"{strike:g}" if isinstance(strike, (int, float)) else str(strike)

            # Find the option contract
            get_options_request = GetOptionContractsRequest(
                underlying_symbols=[symbol.upper()],
                status=AssetStatus.ACTIVE,
                type=ContractType.PUT if contract_type.lower() == 'put' else ContractType.CALL,
                expiration_date_gte=expiration_date,
                expiration_date_lte=expiration_date,
                strike_price_gte=strike_str,
                strike_price_lte=strike_str,
            )
            
            option_contracts_response = await asyncio.to_thread(trading_client.get_option_contracts, get_options_request)
            option_contracts = option_contracts_response.option_contracts


            if not option_contracts:
                return {"error": f"No {contract_type} option contracts found for {symbol} with strike {strike} and expiration {expiration_date}."}
                
            selected_contract = option_contracts[0]
            
            # Place a market order for the option
            order_request = MarketOrderRequest(
                symbol=selected_contract.symbol,
                qty=quantity,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
            )
            
            submitted_order = await asyncio.to_thread(trading_client.submit_order, order_request)
            
            result = {
                "status": "success",
                "message": f"{contract_type.capitalize()} option order placed successfully",
                "order_id": str(submitted_order.id) if getattr(submitted_order, "id", None) is not None else None,
                "symbol": submitted_order.symbol,
                "shares": submitted_order.qty,
                "order_type": "market",
                "timestamp": submitted_order.submitted_at.isoformat(),
            }
            if self.memory:
                self.memory.log_trade(
                    {
                        "symbol": symbol,
                        "qty": float(quantity),
                        "price": float(getattr(submitted_order, "filled_avg_price", 0.0) or 0.0),
                        "side": "buy",
                        "order_id": str(submitted_order.id) if getattr(submitted_order, "id", None) is not None else None,
                        "reason": reason or "Options trade",
                    }
                )
            
            self._log_tool_usage("buy_option", args, result, time.time() - start_time)
            return result

        except Exception as e:
            logging.error(f"Error buying {contract_type} option for {symbol}: {e}")
            result = {"status": "error", "message": str(e)}
            self._log_tool_usage("buy_option", args, result, time.time() - start_time)
            return result
        
    async def modify_trade_parameters(self, order_id: str, new_stop_price: float = None, 
                              new_limit_price: float = None, trail_percent: float = None) -> Dict:
        """
        Modify existing order parameters.
        
        Args:
            order_id: Order ID to modify
            new_stop_price: New stop price (optional)
            new_limit_price: New limit price (optional)
            trail_percent: Trailing stop percentage (optional)
            
        Returns:
            Modification result
        
        Example Tool Input:
            {
                "tool_name": "modify_trade_parameters",
                "args": {
                    "order_id": "12345",
                    "new_stop_price": 470.0,
                    "new_limit_price": 510.0
                }
            }
        
        Example Output:
            {
                "status": "success",
                "new_order_id": "12346"
            }
        """
        import time
        start_time = time.time()
        args = {"order_id": order_id, "new_stop_price": new_stop_price, "new_limit_price": new_limit_price, "trail_percent": trail_percent}
        try:
            # For simplicity, cancel existing order and create new one
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, lambda: self._alpaca_cancel_order(order_id))
            
            if new_stop_price:
                # Get order details to recreate
                order = await loop.run_in_executor(None, self.clients["alpaca"].get_order, order_id)
                
                order_request = StopOrderRequest(
                    symbol=order.symbol,
                    qty=float(order.qty),
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.GTC,
                    stop_price=new_stop_price,
                )
                new_order = await loop.run_in_executor(
                    None,
                    lambda: self.clients["alpaca"].submit_order(order_request),
                )
                result = {"status": "success", "new_order_id": str(new_order.id) if getattr(new_order, "id", None) is not None else None}
            else:
                result = {"status": "success", "message": "Order cancelled"}
            
            self._log_tool_usage("modify_trade_parameters", args, result, time.time() - start_time)
            return result
            
        except Exception as e:
            result = {"status": "error", "message": str(e)}
            self._log_tool_usage("modify_trade_parameters", args, result, time.time() - start_time)
            return result

    async def close_position(self, symbol: str, reason: str = "", shares: float = None) -> Dict:
        """
        Close a position.
        
        Args:
            symbol: Stock symbol to close
            reason: Reason for closing
            shares: Number of shares to close (optional, closes all if not specified)
        
        Returns:
            Position closure result
        
        Example Tool Input:
            {
                "tool_name": "close_position",
                "args": {
                    "symbol": "NVDA",
                    "reason": "Stop loss triggered",
                    "shares": 25
                }
            }
        
        Example Output:
            {
                "status": "success",
                "symbol": "NVDA",
                "shares_closed": 25,
                "order_id": "12345"
            }
        """
        import time
        start_time = time.time()
        args = {"symbol": symbol, "reason": reason, "shares": shares}
        
        print(f"[TRADE] Starting close_position for {symbol}: {shares} shares")
        
        try:
            loop = asyncio.get_running_loop()
            
            # First, cancel any pending orders for this symbol
            print(f"[TRADE] Cancelling pending orders for {symbol}")
            orders = await loop.run_in_executor(None, self._alpaca_list_orders)
            for order in orders:
                if order.symbol == symbol:
                    try:
                        await loop.run_in_executor(None, lambda: self._alpaca_cancel_order(order.id))
                        print(f"[TRADE] Cancelled order {order.id} for {symbol}")
                    except Exception as e:
                        print(f"[TRADE] Error cancelling order {order.id}: {e}")
            
            # Wait for orders to cancel
            await asyncio.sleep(1)
            
            # Check current position
            positions = await loop.run_in_executor(None, self._alpaca_list_positions)
            symbol_position = next((p for p in positions if p.symbol == symbol), None)
            
            if not symbol_position:
                result = {"status": "error", "message": f"No position found for {symbol}"}
                print(f"[TRADE] No position found for {symbol}")
            else:
                available_shares = float(symbol_position.qty)
                print(f"[TRADE] Found {available_shares} shares of {symbol}")
                
                if shares is not None and shares > 0:
                    # Close specific number of shares
                    shares_to_close = min(shares, available_shares)
                    print(f"[TRADE] Closing {shares_to_close} shares of {symbol}")
                    
                    order_request = MarketOrderRequest(
                        symbol=symbol,
                        qty=shares_to_close,
                        side=OrderSide.SELL,
                        type=OrderType.MARKET,
                        time_in_force=TimeInForce.GTC,
                    )
                    order = await loop.run_in_executor(
                        None,
                        lambda: self.clients["alpaca"].submit_order(order_request),
                    )
                    result = {
                        "status": "success", 
                        "symbol": symbol, 
                        "shares_closed": shares_to_close, 
                        "reason": reason, 
                        "order_id": str(order.id) if getattr(order, "id", None) is not None else None,
                        "message": f"Closed {shares_to_close} shares of {symbol}"
                    }
                else:
                    # Close entire position
                    print(f"[TRADE] Closing entire position for {symbol}")
                    await loop.run_in_executor(self.executor, lambda: self.clients["alpaca"].close_position(symbol))
                    result = {
                        "status": "success", 
                        "symbol": symbol, 
                        "shares_closed": available_shares, 
                        "reason": reason, 
                        "message": f"Closed entire position of {available_shares} shares"
                    }
                
                print(f"[TRADE] Close position result: {result}")
            
            self._log_tool_usage("close_position", args, result, time.time() - start_time)
            return result
            
        except Exception as e:
            print(f"[TRADE] ERROR in close_position for {symbol}: {e}")
            result = {"status": "error", "message": str(e)}
            self._log_tool_usage("close_position", args, result, time.time() - start_time)
            return result

    async def add_to_watchlist(self, symbol: str, reason: str = "", technicals: dict = None) -> dict:
        """
        Add symbol to watchlist.
        
        Args:
            symbol: Stock symbol to add
            reason: Optional reason for adding
            technicals: Optional technical analysis snapshot (stored as part of reason if provided)
        
        Returns:
            Watchlist addition result
        
        Example Tool Input:
            {
                "tool_name": "add_to_watchlist",
                "args": {"symbol": "AMD", "reason": "RSI holding 50, base forming"}
            }
        
        Example Output:
            {
                "status": "success",
                "symbol": "AMD",
                "message": "Added to watchlist"
            }
        """
        logging.debug(f"[TOOLBOX] add_to_watchlist called with symbol={symbol}, reason={reason}, memory={self.memory}")
        import time
        start_time = time.time()
        args = {"symbol": symbol, "reason": reason, "technicals": technicals}
        try:
            if not self.memory:
                result = {"error": "Memory system not available"}
            else:
                if technicals and not reason:
                    reason = f"Technicals: {json.dumps(technicals)}"
                self.memory.add_to_watchlist(symbol, reason)
                result = {"status": "success", "symbol": symbol, "message": "Added to watchlist"}
            self._log_tool_usage("add_to_watchlist", args, result, time.time() - start_time)
            return result
        except Exception as e:
            result = {"error": str(e)}
            self._log_tool_usage("add_to_watchlist", args, result, time.time() - start_time)
            return result

    async def get_watchlist(self) -> list:
        """
        Get current watchlist with fresh technicals.
        
        Returns:
            List of watchlist entries
        
        Example Tool Input:
            {
                "tool_name": "get_watchlist",
                "args": {}
            }
        
        Example Output:
            [
                {
                    "symbol": "AMD",
                    "added_at": "2024-01-15T10:30:00Z",
                    "technicals": {
                        "rsi": 65,
                        "trend": "uptrend",
                        "price": 125.50
                    }
                }
            ]
        """
        logging.debug(f"[TOOLBOX] get_watchlist called, memory={self.memory}")
        import time
        start_time = time.time()
        args = {}
        try:
            if not self.memory:
                result = []
            else:
                entries = self.memory.get_watchlist()
                if not entries:
                    result = []
                else:
                    tasks = []
                    index_map = []
                    for entry in entries:
                        symbol = entry.get("symbol")
                        if not symbol:
                            continue
                        index_map.append(entry)
                        tasks.append(self.analyze_technicals(symbol))
                    tech_results = await asyncio.gather(*tasks, return_exceptions=True)
                    result = []
                    for entry, tech in zip(index_map, tech_results):
                        if isinstance(tech, Exception):
                            tech = {"error": str(tech)}
                        result.append(
                            {
                                "symbol": entry.get("symbol"),
                                "added_at": entry.get("added_at"),
                                "reason": entry.get("reason", ""),
                                "technicals": tech,
                            }
                        )
            self._log_tool_usage("get_watchlist", args, result, time.time() - start_time)
            return result
        except Exception as e:
            result = []
            logging.error(f"get_watchlist error: {e}")
            self._log_tool_usage("get_watchlist", args, result, time.time() - start_time)
            return result

    async def remove_from_watchlist(self, symbol: str) -> dict:
        """
        Remove symbol from watchlist.
        
        Args:
            symbol: Stock symbol to remove
        
        Returns:
            Watchlist removal result
        
        Example Tool Input:
            {
                "tool_name": "remove_from_watchlist",
                "args": {"symbol": "AMD"}
            }
        
        Example Output:
            {
                "status": "success",
                "symbol": "AMD",
                "message": "Removed from watchlist"
            }
        """
        logging.debug(f"[TOOLBOX] remove_from_watchlist called with symbol={symbol}, memory={self.memory}")
        import time
        start_time = time.time()
        args = {"symbol": symbol}
        try:
            if not self.memory:
                result = {"error": "Memory system not available"}
            else:
                result = self.memory.remove_from_watchlist(symbol)
            self._log_tool_usage("remove_from_watchlist", args, result, time.time() - start_time)
            return result
        except Exception as e:
            result = {"error": str(e)}
            self._log_tool_usage("remove_from_watchlist", args, result, time.time() - start_time)
            return result

    async def save_notes(self, notes: str) -> dict:
        """Save notes to notes.md with a sliding window of 10 entries."""
        try:
            note_text = str(notes).strip()
            if not note_text:
                return {"status": "error", "message": "Notes cannot be empty"}

            note_text = " ".join(note_text.split())
            existing = self._load_notes()
            existing.append(
                {"timestamp": dt.datetime.now(dt.timezone.utc).isoformat(), "note": note_text}
            )
            if len(existing) > 10:
                existing = existing[-10:]
            self._write_notes(existing)
            return {
                "status": "success",
                "message": "Note saved",
                "path": self._notes_path(),
                "saved_note": existing[-1],
                "count": len(existing),
            }
        except Exception as e:
            return {"status": "error", "message": f"Failed to save notes: {str(e)}"}

    async def return_research_results(self, controlling_narrative: str, candidates: list) -> dict:
        """A dummy tool used by the orchestrator to signal the end of the research phase."""
        return {
            "status": "success", 
            "data": {
                "controlling_narrative": controlling_narrative, 
                "candidates": candidates
            }
        }

    async def pass_on_trades(self, reason: str) -> dict:
        """A dummy tool used by the orchestrator to signal that no trades were taken."""
        return {
            "status": "success", 
            "reason": reason
        }

    async def read_notes(self) -> dict:
        """Read notes from notes.md."""
        notes = self._load_notes()
        return {"notes": notes, "path": self._notes_path(), "count": len(notes)}

    @cache(ttl=86400)  # Cache the list for 24 hours (86400 seconds)
    def get_sp500_symbols(self) -> List[str]:
        """
        Gets the list of S&P 500 symbols from Wikipedia.
        Results are cached for 24 hours to prevent excessive scraping.
        """
        try:
            logging.info("Fetching S&P 500 symbol list from Wikipedia.")
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            # The [0] is to select the first table found on the page
            table = pd.read_html(url, header=0)[0]
            symbols = table['Symbol'].tolist()
            # Clean up symbols that might have different representations (e.g., 'BRK.B' -> 'BRK-B')
            symbols = [s.replace('.', '-') for s in symbols]
            logging.info(f"Successfully fetched {len(symbols)} S&P 500 symbols.")
            return symbols
        except Exception as e:
            logging.error(f"Could not fetch S&P 500 symbol list: {e}")
            # Fallback to a manually curated list of major stocks in case of failure
            return ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA", "JPM", "JNJ", "V"]

    async def get_company_news(self, symbol: str, from_date: str, to_date: str) -> List[Dict]:
        """
        Fetch company news for a given symbol and date range from Finnhub.
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)
        Returns:
            List of news articles for the company
        Example Tool Input:
            {
                "tool_name": "get_company_news",
                "args": {"symbol": "AAPL", "from_date": "2024-06-01", "to_date": "2024-06-22"}
            }
        Example Output:
            [
                {
                    "category": "company",
                    "datetime": 1705344000,
                    "headline": "Apple launches new product...",
                    "id": 12345,
                    "image": "https://example.com/image.jpg",
                    "related": "AAPL",
                    "source": "Reuters",
                    "summary": "Apple announced...",
                    "url": "https://example.com/article"
                }
            ]
        """
        import time
        start_time = time.time()
        async with self.semaphores['finnhub']:
            await self._rate_limit('finnhub')
            try:
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: requests.get(
                        "https://finnhub.io/api/v1/company-news",
                        params={
                            "symbol": symbol,
                            "from": from_date,
                            "to": to_date,
                            "token": self.clients["finnhub_key"]
                        },
                        timeout=10
                    )
                )
                result = response.json()[:25] if response.status_code == 200 else []
                self._log_tool_usage("get_company_news", {"symbol": symbol, "from_date": from_date, "to_date": to_date}, result, time.time() - start_time)
                return result
            except Exception as e:
                import logging
                logging.error(f"Error fetching company news for {symbol}: {e}")
                result = []
                self._log_tool_usage("get_company_news", {"symbol": symbol, "from_date": from_date, "to_date": to_date}, {"error": str(e)}, time.time() - start_time)
                return result
