import asyncio
import logging
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Any
import yfinance as yf


class EarningsCalendar:
    """Earnings calendar integration using Finnhub and fallbacks."""
    
    def __init__(self, clients: Dict):
        self.clients = clients
        self.finnhub_key = clients.get("finnhub_key")
    
    async def get_earnings_calendar(
        self, 
        start_date: Any = None,
        end_date: Any = None,
        limit: int = 100
    ) -> List[Dict]:
        """
        Get earnings calendar for specified date range.
        
        Args:
            start_date: Start date in YYYY-MM-DD format (default: today)
            end_date: End date in YYYY-MM-DD format (default: +7 days)
            limit: Maximum number of results
            
        Returns:
            List of earnings events with symbol, date, time, eps_estimate, etc.
            
        Example Usage:
            # Get earnings for next 7 days
            earnings = await calendar.get_earnings_calendar()
            
            # Get earnings for specific date range
            earnings = await calendar.get_earnings_calendar(
                start_date="2024-01-15",
                end_date="2024-01-22",
                limit=50
            )
            
        Example Output:
            [
                {
                    "symbol": "AAPL",
                    "date": "2024-01-18",
                    "hour": "16:30",
                    "quarter": 1,
                    "year": 2024,
                    "eps_estimate": 2.10,
                    "eps_actual": None,
                    "revenue_estimate": 117000000000,
                    "revenue_actual": None,
                    "source": "finnhub"
                }
            ]
        """
        try:
            # Set default dates
            if not start_date:
                start_date = datetime.now().strftime("%Y-%m-%d")
            if not end_date:
                end_date = (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")
            
            # Try Finnhub first
            if self.finnhub_key:
                result = await self._get_finnhub_earnings(start_date, end_date, limit)
                if result:
                    return result
            
            # Fallback to Yahoo Finance
            result = await self._get_yahoo_earnings(start_date, end_date, limit)
            return result
            
        except Exception as e:
            logging.error(f"Error getting earnings calendar: {e}")
            return []
    
    async def _get_finnhub_earnings(self, start_date: str, end_date: str, limit: int) -> List[Dict]:
        """Get earnings from Finnhub API."""
        try:
            url = "https://finnhub.io/api/v1/calendar/earnings"
            params = {
                "token": self.finnhub_key,
                "from": start_date,
                "to": end_date
            }
            
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: requests.get(url, params=params, timeout=15)
            )
            
            if response.status_code == 200:
                data = response.json()
                earnings = []
                
                for event in data.get("earningsCalendar", [])[:limit]:
                    earnings.append({
                        "symbol": event.get("symbol", ""),
                        "date": event.get("date", ""),
                        "hour": event.get("hour", ""),
                        "quarter": event.get("quarter", 0),
                        "year": event.get("year", 0),
                        "eps_estimate": float(event.get("epsEstimate", 0)),
                        "eps_actual": float(event.get("epsActual", 0)) if event.get("epsActual") else None,
                        "revenue_estimate": float(event.get("revenueEstimate", 0)),
                        "revenue_actual": float(event.get("revenueActual", 0)) if event.get("revenueActual") else None,
                        "source": "finnhub"
                    })
                
                return earnings
            else:
                logging.warning(f"Finnhub earnings API returned {response.status_code}")
                return []
                
        except Exception as e:
            logging.error(f"Error with Finnhub earnings: {e}")
            return []
    
    async def _get_yahoo_earnings(self, start_date: str, end_date: str, limit: int) -> List[Dict]:
        """Fallback earnings using Yahoo Finance for popular symbols."""
        try:
            # Popular symbols likely to have earnings
            symbols = [
                "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "NFLX",
                "AMD", "INTC", "PYPL", "ADBE", "CRM", "UBER", "SHOP", "SQ",
                "ROKU", "SNOW", "PLTR", "RBLX", "COIN", "HOOD", "SOFI", "UPST"
            ]
            
            earnings = []
            
            # Process symbols in batches
            batch_size = 5
            for i in range(0, len(symbols), batch_size):
                batch = symbols[i:i + batch_size]
                
                tasks = []
                for symbol in batch:
                    tasks.append(self._get_symbol_earnings(symbol, start_date, end_date))
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for result in results:
                    if isinstance(result, dict) and result:
                        earnings.append(result)
                
                # Small delay between batches
                await asyncio.sleep(0.2)
                
                if len(earnings) >= limit:
                    break
            
            return earnings[:limit]
            
        except Exception as e:
            logging.error(f"Error with Yahoo earnings fallback: {e}")
            return []
    
    async def _get_symbol_earnings(self, symbol: str, start_date: str, end_date: str) -> Dict:
        """Get earnings data for a single symbol."""
        try:
            loop = asyncio.get_event_loop()
            
            ticker = yf.Ticker(symbol)
            
            # Get calendar data
            calendar = await loop.run_in_executor(None, ticker.calendar)
            
            if calendar is not None and not calendar.empty:
                # Get the next earnings date
                earnings_date = calendar.index[0] if len(calendar) > 0 else None
                
                if earnings_date:
                    earnings_date_str = earnings_date.strftime("%Y-%m-%d")
                    
                    # Check if date is in range
                    if start_date <= earnings_date_str <= end_date:
                        # Get EPS estimate
                        eps_estimate = calendar.iloc[0, 0] if len(calendar.columns) > 0 else 0
                        
                        return {
                            "symbol": symbol,
                            "date": earnings_date_str,
                            "hour": "unknown",
                            "quarter": 0,
                            "year": earnings_date.year,
                            "eps_estimate": float(eps_estimate) if eps_estimate else 0.0,
                            "eps_actual": None,
                            "revenue_estimate": 0.0,
                            "revenue_actual": None,
                            "source": "yahoo"
                        }
            
            return {}
            
        except Exception as e:
            logging.warning(f"Error getting earnings for {symbol}: {e}")
            return {}
    
    async def get_earnings_for_symbols(self, symbols: List[str]) -> List[Dict]:
        """
        Get earnings data for specific symbols.
        
        Args:
            symbols: List of stock symbols to check
            
        Returns:
            List of detailed earnings data for each symbol
            
        Example Usage:
            # Get earnings for specific symbols
            earnings = await calendar.get_earnings_for_symbols(["AAPL", "MSFT", "GOOGL"])
            
        Example Output:
            [
                {
                    "symbol": "AAPL",
                    "next_earnings_date": "2024-01-18",
                    "eps_estimate": 2.10,
                    "historical_eps": [1.88, 1.52, 1.29, 1.20],
                    "eps_growth": 23.7,
                    "forward_pe": 28.5,
                    "trailing_pe": 30.2,
                    "peg_ratio": 1.8,
                    "earnings_growth": 0.15
                }
            ]
        """
        try:
            earnings = []
            
            # Process symbols in batches
            batch_size = 5
            for i in range(0, len(symbols), batch_size):
                batch = symbols[i:i + batch_size]
                
                tasks = []
                for symbol in batch:
                    tasks.append(self._get_detailed_earnings(symbol))
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for result in results:
                    if isinstance(result, dict) and result:
                        earnings.append(result)
                
                # Small delay between batches
                await asyncio.sleep(0.2)
            
            return earnings
            
        except Exception as e:
            logging.error(f"Error getting earnings for symbols: {e}")
            return []
    
    async def _get_detailed_earnings(self, symbol: str) -> Dict:
        """Get detailed earnings data for a symbol."""
        try:
            loop = asyncio.get_event_loop()
            
            ticker = yf.Ticker(symbol)
            
            # Get multiple data sources
            tasks = [
                loop.run_in_executor(None, getattr, ticker, 'calendar'),
                loop.run_in_executor(None, getattr, ticker, 'earnings'),
                loop.run_in_executor(None, getattr, ticker, 'info')
            ]
            
            calendar, earnings, info = await asyncio.gather(*tasks, return_exceptions=True)
            
            result = {"symbol": symbol}
            
            # Process calendar data
            if isinstance(calendar, Exception):
                calendar = None
            if calendar is not None and not calendar.empty:
                earnings_date = calendar.index[0]
                result["next_earnings_date"] = earnings_date.strftime("%Y-%m-%d")
                if len(calendar.columns) > 0:
                    result["eps_estimate"] = float(calendar.iloc[0, 0]) if calendar.iloc[0, 0] else 0.0
            
            # Process historical earnings
            if isinstance(earnings, Exception):
                earnings = None
            if earnings is not None and not earnings.empty:
                # Get last 4 quarters
                recent_earnings = earnings.tail(4)
                result["historical_eps"] = recent_earnings.tolist()
                
                # Calculate earnings growth
                if len(recent_earnings) >= 2:
                    current_eps = recent_earnings.iloc[-1]
                    previous_eps = recent_earnings.iloc[-2]
                    if previous_eps != 0:
                        growth = ((current_eps - previous_eps) / abs(previous_eps)) * 100
                        result["eps_growth"] = float(growth)
            
            # Process info data
            if isinstance(info, Exception):
                info = None
            if info and isinstance(info, dict):
                result["forward_pe"] = info.get("forwardPE")
                result["trailing_pe"] = info.get("trailingPE")
                result["peg_ratio"] = info.get("pegRatio")
                result["earnings_growth"] = info.get("earningsGrowth")
            
            return result if len(result) > 1 else {}
            
        except Exception as e:
            logging.warning(f"Error getting detailed earnings for {symbol}: {e}")
            return {}
    
    async def get_earnings_surprises(self, symbols: List[str]) -> List[Dict]:
        """
        Get recent earnings surprises for symbols.
        
        Args:
            symbols: List of stock symbols to check
            
        Returns:
            List of earnings surprise data
            
        Example Usage:
            # Get earnings surprises for symbols
            surprises = await calendar.get_earnings_surprises(["AAPL", "MSFT", "TSLA"])
            
        Example Output:
            [
                {
                    "symbol": "AAPL",
                    "date": "2023-10-26",
                    "actual": 1.46,
                    "estimate": 1.39,
                    "surprise": 0.07,
                    "surprise_pct": 5.04,
                    "quarter": 4,
                    "year": 2023
                }
            ]
        """
        try:
            surprises = []
            
            for symbol in symbols:
                try:
                    if self.finnhub_key:
                        surprise = await self._get_finnhub_surprise(symbol)
                        if surprise:
                            surprises.append(surprise)
                    
                    # Small delay between requests
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    logging.warning(f"Error getting surprise for {symbol}: {e}")
                    continue
            
            return surprises
            
        except Exception as e:
            logging.error(f"Error getting earnings surprises: {e}")
            return []
    
    async def _get_finnhub_surprise(self, symbol: str) -> Any:
        """Get earnings surprise from Finnhub."""
        try:
            url = "https://finnhub.io/api/v1/stock/earnings"
            params = {
                "symbol": symbol,
                "token": self.finnhub_key
            }
            
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: requests.get(url, params=params, timeout=10)
            )
            
            if response.status_code == 200:
                data = response.json()
                
                if data and len(data) > 0:
                    latest = data[0]  # Most recent earnings
                    
                    actual = latest.get("actual")
                    estimate = latest.get("estimate")
                    
                    if actual is not None and estimate is not None:
                        surprise = actual - estimate
                        surprise_pct = (surprise / abs(estimate)) * 100 if estimate != 0 else 0
                        
                        return {
                            "symbol": symbol,
                            "date": latest.get("date"),
                            "actual": float(actual),
                            "estimate": float(estimate),
                            "surprise": float(surprise),
                            "surprise_pct": float(surprise_pct),
                            "quarter": latest.get("quarter"),
                            "year": latest.get("year")
                        }
            
            return None
            
        except Exception as e:
            logging.warning(f"Error getting Finnhub surprise for {symbol}: {e}")
            return None