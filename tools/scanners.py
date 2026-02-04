import asyncio
import logging
import requests
import yfinance as yf
import pandas as pd
from alpaca.data.requests import StockSnapshotRequest
from datetime import datetime
from typing import Dict, List

class MarketScanner:
    """Fast market scanners for movers and unusual volume."""

    def __init__(self, clients: Dict):
        self.clients = clients
        self.alpaca_client = clients.get("alpaca")
        self.alpaca_data_client = clients.get("alpaca_data")
        self.finnhub_key = clients.get("finnhub_key")

    async def scan_top_movers(self, limit: int = 50) -> List[Dict]:
        """
        Scan top movers using Alpaca snapshots or Finnhub gainers.

        Args:
            limit: Number of top movers to return

        Returns:
            List of dicts with symbol, price, change, change_pct, volume

        Example Usage:
            # Get top 50 movers
            movers = await scanner.scan_top_movers()

            # Get top 20 movers
            movers = await scanner.scan_top_movers(limit=20)

        Example Output:
            [
                {
                    "symbol": "NVDA",
                    "price": 485.50,
                    "change": 15.75,
                    "change_pct": 3.35,
                    "volume": 45678900,
                    "high": 487.25,
                    "low": 472.10,
                    "timestamp": "2024-01-15T16:00:00Z"
                }
            ]
        """
        try:
            # Try Alpaca first
            if self.alpaca_client:
                result = await self._scan_alpaca_snapshots(limit)
                if result:
                    return result

            # Fallback to Finnhub
            if self.finnhub_key:
                result = await self._scan_finnhub_gainers(limit)
                if result:
                    return result

            # Last resort: scan popular symbols
            return await self._scan_popular_symbols(limit)

        except Exception as e:
            logging.error(f"Error scanning top movers: {e}")
            return []

    async def _scan_alpaca_snapshots(self, limit: int) -> List[Dict]:
        """Scan using Alpaca snapshots API."""
        try:
            if not self.alpaca_data_client:
                return []

            # Get list of popular symbols to scan
            symbols = [
                "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "NFLX",
                "AMD", "INTC", "BABA", "CRM", "UBER", "SHOP", "SQ", "ROKU",
                "PYPL", "ADBE", "ORCL", "QCOM", "AMAT", "MU", "LRCX", "KLAC",
                "MRVL", "MCHP", "XLNX", "SNPS", "CDNS", "INTU", "WDAY", "TEAM",
                "ZM", "DOCU", "CRWD", "OKTA", "SPLK", "SNOW", "PLTR", "RBLX",
                "COIN", "HOOD", "SOFI", "UPST", "AFRM", "PATH", "DDOG", "MDB"
            ]

            # Get snapshots
            loop = asyncio.get_event_loop()
            snapshots = await loop.run_in_executor(
                None,
                lambda: self.alpaca_data_client.get_stock_snapshot(
                    StockSnapshotRequest(symbol_or_symbols=symbols)
                )
            )

            movers = []
            for symbol, snapshot in snapshots.items():
                if not snapshot or not getattr(snapshot, "daily_bar", None):
                    continue

                bar = snapshot.daily_bar
                prev_bar = getattr(snapshot, "previous_daily_bar", None)
                prev_close = getattr(prev_bar, "close", None)
                if prev_close is None:
                    prev_close = getattr(bar, "open", None)

                if prev_close and prev_close > 0:
                    change = bar.close - prev_close
                    change_pct = (change / prev_close) * 100
                else:
                    change = 0.0
                    change_pct = 0.0

                movers.append({
                    "symbol": symbol,
                    "price": float(bar.close),
                    "change": float(change),
                    "change_pct": float(change_pct),
                    "volume": int(bar.volume),
                    "high": float(bar.high),
                    "low": float(bar.low),
                    "timestamp": bar.timestamp.isoformat()
                })

            # Sort by absolute change percentage
            movers.sort(key=lambda x: abs(x["change_pct"]), reverse=True)
            return movers[:limit]

        except Exception as e:
            logging.error(f"Error with Alpaca snapshots: {e}")
            return []

    async def _scan_finnhub_gainers(self, limit: int) -> List[Dict]:
        """Scan using Finnhub top gainers API."""
        try:
            url = "https://finnhub.io/api/v1/stock/top-gainers"
            params = {"token": self.finnhub_key}

            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: requests.get(url, params=params, timeout=10)
            )

            if response.status_code == 200:
                data = response.json()
                movers = []

                for item in data.get("data", [])[:limit]:
                    movers.append({
                        "symbol": item.get("symbol", ""),
                        "price": float(item.get("price", 0)),
                        "change": float(item.get("change", 0)),
                        "change_pct": float(item.get("changesPercentage", 0)),
                        "volume": int(item.get("volume", 0)),
                        "source": "finnhub_gainers"
                    })

                return movers

        except Exception as e:
            logging.error(f"Error with Finnhub gainers: {e}")
            return []

    async def _scan_popular_symbols(self, limit: int) -> List[Dict]:
        """Fallback scan of popular symbols using yfinance."""
        try:
            symbols = [
                "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "NFLX",
                "AMD", "INTC", "PYPL", "ADBE", "CRM", "UBER", "SHOP", "SQ"
            ]

            loop = asyncio.get_event_loop()

            # Get current data for symbols
            movers = []
            try:
                df = await loop.run_in_executor(
                    None,
                    lambda: yf.download(
                        tickers=symbols[:limit],
                        period="2d",
                        interval="1d",
                        progress=False,
                        auto_adjust=False,
                    )
                )
            except Exception as e:
                logging.warning(f"Error downloading yfinance data: {e}")
                return []

            if df.empty:
                return []

            if isinstance(df.columns, pd.MultiIndex):
                close_df = df["Close"] if "Close" in df else None
                volume_df = df["Volume"] if "Volume" in df else None
                if close_df is None:
                    return []

                for symbol in symbols[:limit]:
                    if symbol not in close_df:
                        continue
                    close_series = close_df[symbol].dropna()
                    if len(close_series) < 2:
                        continue
                    prev_close = float(close_series.iloc[-2])
                    price = float(close_series.iloc[-1])
                    if prev_close <= 0:
                        continue
                    change = price - prev_close
                    change_pct = (change / prev_close) * 100
                    volume = int(volume_df[symbol].iloc[-1]) if volume_df is not None and symbol in volume_df else 0

                    movers.append({
                        "symbol": symbol,
                        "price": price,
                        "change": float(change),
                        "change_pct": float(change_pct),
                        "volume": volume,
                        "source": "yfinance_download"
                    })
            else:
                close_series = df["Close"].dropna() if "Close" in df else pd.Series(dtype=float)
                if len(close_series) >= 2:
                    prev_close = float(close_series.iloc[-2])
                    price = float(close_series.iloc[-1])
                    if prev_close > 0:
                        change = price - prev_close
                        change_pct = (change / prev_close) * 100
                        volume = int(df["Volume"].iloc[-1]) if "Volume" in df else 0
                        movers.append({
                            "symbol": symbols[0],
                            "price": price,
                            "change": float(change),
                            "change_pct": float(change_pct),
                            "volume": volume,
                            "source": "yfinance_download"
                        })

            # Sort by absolute change percentage
            movers.sort(key=lambda x: abs(x["change_pct"]), reverse=True)
            return movers

        except Exception as e:
            logging.error(f"Error with popular symbols scan: {e}")
            return []

    async def scan_unusual_volume(self, threshold: float = 2.0, limit: int = 50) -> List[Dict]:
        """
        Scan for unusual volume using relative volume ratio.

        Args:
            threshold: Minimum relative volume ratio (current vs 20-day avg)
            limit: Number of results to return

        Returns:
            List of dicts with symbol, volume, avg_volume, volume_ratio, price

        Example Usage:
            # Find stocks with 2x average volume
            unusual = await scanner.scan_unusual_volume(threshold=2.0)

            # Find stocks with 3x average volume, top 20
            unusual = await scanner.scan_unusual_volume(threshold=3.0, limit=20)

        Example Output:
            [
                {
                    "symbol": "TSLA",
                    "volume": 125000000,
                    "avg_volume_20d": 45678900,
                    "volume_ratio": 2.74,
                    "price": 245.50,
                    "threshold_met": True
                }
            ]
        """
        try:
            # Get list of symbols to scan
            symbols = [
                "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "NFLX",
                "AMD", "INTC", "PYPL", "ADBE", "CRM", "UBER", "SHOP", "SQ",
                "ROKU", "TWTR", "SNAP", "PINS", "DOCU", "ZM", "CRWD", "OKTA",
                "SNOW", "PLTR", "RBLX", "COIN", "HOOD", "SOFI", "UPST", "AFRM"
            ]

            unusual_volume = []

            # Process symbols in batches to avoid rate limits
            batch_size = 10
            for i in range(0, len(symbols), batch_size):
                batch = symbols[i:i + batch_size]

                tasks = []
                for symbol in batch:
                    tasks.append(self._get_volume_data(symbol, threshold))

                results = await asyncio.gather(*tasks, return_exceptions=True)

                for result in results:
                    if isinstance(result, dict) and result.get("volume_ratio", 0) >= threshold:
                        unusual_volume.append(result)

                # Small delay between batches
                await asyncio.sleep(0.1)

            # Sort by volume ratio descending
            unusual_volume.sort(key=lambda x: x.get("volume_ratio", 0), reverse=True)
            return unusual_volume[:limit]

        except Exception as e:
            logging.error(f"Error scanning unusual volume: {e}")
            return []

    async def _get_volume_data(self, symbol: str, threshold: float) -> Dict:
        """Get volume data for a single symbol."""
        try:
            loop = asyncio.get_event_loop()

            # Get 30 days of data to calculate average
            ticker = yf.Ticker(symbol)
            hist = await loop.run_in_executor(
                None,
                lambda: ticker.history(period="30d", interval="1d")
            )

            if hist.empty or len(hist) < 20:
                return {}

            # Calculate metrics
            current_volume = hist['Volume'].iloc[-1]
            avg_volume_20d = hist['Volume'].iloc[-20:].mean()
            current_price = hist['Close'].iloc[-1]

            volume_ratio = current_volume / avg_volume_20d if avg_volume_20d > 0 else 0

            if volume_ratio >= threshold:
                return {
                    "symbol": symbol,
                    "volume": int(current_volume),
                    "avg_volume_20d": int(avg_volume_20d),
                    "volume_ratio": float(volume_ratio),
                    "price": float(current_price),
                    "threshold_met": True
                }

            return {}

        except Exception as e:
            logging.warning(f"Error getting volume data for {symbol}: {e}")
            return {}

    async def scan_premarket_movers(self, limit: int = 20) -> List[Dict]:
        """
        Scan premarket movers using Finnhub.

        Args:
            limit: Number of premarket movers to return

        Returns:
            List of premarket movers with price and change data

        Example Usage:
            # Get premarket movers
            premarket = await scanner.scan_premarket_movers()

            # Get top 10 premarket movers
            premarket = await scanner.scan_premarket_movers(limit=10)

        Example Output:
            [
                {
                    "symbol": "NVDA",
                    "price": 485.50,
                    "change": 12.25,
                    "change_pct": 2.59,
                    "volume": 1250000,
                    "market_session": "premarket"
                }
            ]
        """
        try:
            if not self.finnhub_key:
                return []

            url = "https://finnhub.io/api/v1/stock/premarket-movers"
            params = {"token": self.finnhub_key}

            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: requests.get(url, params=params, timeout=10)
            )

            if response.status_code == 200:
                data = response.json()
                movers = []

                for item in data.get("data", [])[:limit]:
                    movers.append({
                        "symbol": item.get("symbol", ""),
                        "price": float(item.get("price", 0)),
                        "change": float(item.get("change", 0)),
                        "change_pct": float(item.get("changesPercentage", 0)),
                        "volume": int(item.get("volume", 0)),
                        "market_session": "premarket"
                    })

                return movers

        except Exception as e:
            logging.error(f"Error scanning premarket movers: {e}")
            return []

    async def get_market_summary(self) -> Dict:
        """
        Get overall market summary.

        Returns:
            Market summary with major indices data

        Example Usage:
            # Get market summary
            summary = await scanner.get_market_summary()

        Example Output:
            {
                "timestamp": "2024-01-15T16:00:00Z",
                "indices": {
                    "SPY": {
                        "price": 485.50,
                        "change": 2.15,
                        "change_pct": 0.44,
                        "volume": 45678900
                    },
                    "QQQ": {
                        "price": 425.75,
                        "change": 3.25,
                        "change_pct": 0.77,
                        "volume": 23456700
                    },
                    "IWM": {
                        "price": 195.25,
                        "change": -0.75,
                        "change_pct": -0.38,
                        "volume": 12345600
                    },
                    "VIX": {
                        "price": 12.85,
                        "change": -0.45,
                        "change_pct": -3.38,
                        "volume": 8900000
                    }
                }
            }
        """
        try:
            # Get major indices
            indices = ["SPY", "QQQ", "IWM", "VIX"]

            tasks = []
            for symbol in indices:
                tasks.append(self._get_index_data(symbol))

            results = await asyncio.gather(*tasks, return_exceptions=True)

            summary = {
                "timestamp": datetime.now().isoformat(),
                "indices": {}
            }

            for i, result in enumerate(results):
                if isinstance(result, dict):
                    summary["indices"][indices[i]] = result

            return summary

        except Exception as e:
            logging.error(f"Error getting market summary: {e}")
            return {"error": str(e)}

    async def _get_index_data(self, symbol: str) -> Dict:
        """Get data for a single index."""
        try:
            loop = asyncio.get_event_loop()

            ticker = yf.Ticker(symbol)
            info = await loop.run_in_executor(None, ticker.info)

            if info:
                price = info.get("regularMarketPrice", 0)
                prev_close = info.get("regularMarketPreviousClose", 0)

                if price > 0 and prev_close > 0:
                    change = price - prev_close
                    change_pct = (change / prev_close) * 100

                    return {
                        "price": float(price),
                        "change": float(change),
                        "change_pct": float(change_pct),
                        "volume": int(info.get("regularMarketVolume", 0))
                    }

            return {}

        except Exception as e:
            logging.warning(f"Error getting index data for {symbol}: {e}")
            return {}
