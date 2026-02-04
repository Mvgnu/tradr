import asyncio
import logging
import time
from typing import Dict, Callable, Optional
from dataclasses import dataclass, field
from enum import Enum


class WatcherStatus(Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class TechnicalWatcher:
    symbol: str
    interval: float  # seconds
    callback: Callable
    condition: Callable[[Dict], bool]
    task: Optional[asyncio.Task] = None
    status: WatcherStatus = WatcherStatus.ACTIVE
    last_check: float = 0
    error_count: int = 0
    created_at: float = field(default_factory=time.time)


class WatcherManager:
    """Manages technical analysis watchers with async polling."""
    
    def __init__(self, toolbox):
        self.toolbox = toolbox
        self.watchers: Dict[str, TechnicalWatcher] = {}
        self.running = False
        self.cleanup_task = None
    
    async def check_technical_async(self, symbol: str) -> bool:
        """
        Async version: always non-blocking.

        Args:
            symbol: Stock symbol to check

        Returns:
            Boolean indicating if technical conditions are met

        Example Usage:
            # Check if AAPL meets technical conditions
            is_bullish = await watcher.check_technical_async("AAPL")

        Example Output:
            True  # If RSI 30-70, MACD bullish, and uptrend
        """
        try:
            result = await self.toolbox.analyze_technicals(symbol)
            if isinstance(result, dict) and "error" not in result:
                rsi = result.get("rsi", 50)
                macd_signal = result.get("macd_signal", "neutral")
                trend = result.get("trend", "sideways")
                return 30 < rsi < 70 and macd_signal == "bullish" and trend == "uptrend"
            return False
        except Exception as e:
            logging.error(f"Error in check_technical_async for {symbol}: {e}")
            return False
    
    async def _watch_symbol(self, watcher: TechnicalWatcher):
        """Internal watcher loop for a single symbol."""
        while watcher.status == WatcherStatus.ACTIVE:
            try:
                # Get technical analysis
                result = await self.toolbox.analyze_technicals(watcher.symbol)
                watcher.last_check = time.time()
                
                if isinstance(result, dict) and "error" not in result:
                    # Check condition
                    if watcher.condition(result):
                        # Condition met, call callback
                        await watcher.callback(watcher.symbol, result)
                    
                    # Reset error count on successful check
                    watcher.error_count = 0
                else:
                    logging.warning(f"Watcher {watcher.symbol}: {result}")
                    watcher.error_count += 1
                
                # Check if too many errors
                if watcher.error_count > 3:
                    watcher.status = WatcherStatus.ERROR
                    logging.error(f"Watcher {watcher.symbol} stopped due to errors")
                    break
                
                await asyncio.sleep(watcher.interval)
                
            except asyncio.CancelledError:
                logging.info(f"Watcher {watcher.symbol} cancelled")
                break
            except Exception as e:
                logging.error(f"Error in watcher {watcher.symbol}: {e}")
                watcher.error_count += 1
                await asyncio.sleep(watcher.interval)
        
        watcher.status = WatcherStatus.STOPPED
    
    def register_watcher(
        self, 
        symbol: str, 
        interval: float = 60.0,
        condition: Optional[Callable[[Dict], bool]] = None,
        callback: Optional[Callable] = None,
    ) -> str:
        """
        Register a new technical watcher.
        
        Args:
            symbol: Symbol to watch
            interval: Polling interval in seconds
            condition: Function that takes technical data and returns bool
            callback: Async function to call when condition is met
        
        Returns:
            Watcher ID (symbol for now)

        Example Usage:
            # Register a basic bullish watcher
            watcher_id = watcher.register_watcher("AAPL", interval=30.0)

            # Register custom condition watcher
            def breakout_condition(data):
                return data.get("volume_trend") == "surge" and data.get("rsi", 50) > 60

            async def breakout_callback(symbol, data):
                print(f"Breakout detected for {symbol}!")

            watcher_id = watcher.register_watcher(
                symbol="NVDA",
                interval=15.0,
                condition=breakout_condition,
                callback=breakout_callback
            )

        Example Output:
            "AAPL"  # Returns the symbol as watcher ID
        """
        if symbol in self.watchers:
            logging.warning(f"Watcher for {symbol} already exists")
            return symbol
        
        # Default condition: bullish signals
        if condition is None:
            def default_condition(data):
                return (
                    data.get("macd_signal") == "bullish"
                    and data.get("trend") == "uptrend"
                    and 30 < data.get("rsi", 50) < 70
            )
            condition = default_condition
        
        # Default callback: log alert
        if callback is None:
            callback = self._default_callback
        
        watcher = TechnicalWatcher(symbol=symbol, interval=interval, callback=callback, condition=condition)
        
        self.watchers[symbol] = watcher
        
        # Start the watcher task
        if self.running:
            watcher.task = asyncio.create_task(self._watch_symbol(watcher))
        
        logging.info(f"Registered watcher for {symbol} with {interval}s interval")
        return symbol
    
    def cancel_watcher(self, symbol: str) -> bool:
        """
        Cancel a watcher.

        Args:
            symbol: Symbol to cancel watcher for

        Returns:
            Boolean indicating if watcher was cancelled

        Example Usage:
            # Cancel watcher for AAPL
            cancelled = watcher.cancel_watcher("AAPL")

        Example Output:
            True  # If watcher was found and cancelled
        """
        if symbol not in self.watchers:
            logging.warning(f"No watcher found for {symbol}")
            return False
        
        watcher = self.watchers[symbol]
        watcher.status = WatcherStatus.STOPPED
        
        if watcher.task and not watcher.task.done():
            watcher.task.cancel()
        
        del self.watchers[symbol]
        logging.info(f"Cancelled watcher for {symbol}")
        return True
    
    def list_watchers(self) -> Dict[str, Dict]:
        """
        List all watchers and their status.

        Returns:
            Dictionary of watcher status information

        Example Usage:
            # Get all active watchers
            watchers = watcher.list_watchers()

        Example Output:
            {
                "AAPL": {
                    "status": "active",
                    "interval": 60.0,
                    "last_check": 1705344000.0,
                    "error_count": 0,
                    "created_at": 1705343940.0,
                    "uptime": 60.0
                },
                "NVDA": {
                    "status": "active",
                    "interval": 30.0,
                    "last_check": 1705344015.0,
                    "error_count": 0,
                    "created_at": 1705343985.0,
                    "uptime": 30.0
                }
            }
        """
        return {
            symbol: {
                "status": watcher.status.value,
                "interval": watcher.interval,
                "last_check": watcher.last_check,
                "error_count": watcher.error_count,
                "created_at": watcher.created_at,
                "uptime": time.time() - watcher.created_at,
            }
            for symbol, watcher in self.watchers.items()
        }
    
    async def start_all_watchers(self):
        """
        Start all registered watchers.

        Example Usage:
            # Start all watchers
            await watcher.start_all_watchers()

        Example Output:
            # Logs: "Started 3 watchers"
        """
        self.running = True
        
        for watcher in self.watchers.values():
            if watcher.status == WatcherStatus.ACTIVE and not watcher.task:
                watcher.task = asyncio.create_task(self._watch_symbol(watcher))
        
        # Start cleanup task
        if not self.cleanup_task:
            self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        logging.info(f"Started {len(self.watchers)} watchers")
    
    async def stop_all_watchers(self):
        """
        Stop all watchers.

        Example Usage:
            # Stop all watchers
            await watcher.stop_all_watchers()

        Example Output:
            # Logs: "Stopped all watchers"
        """
        self.running = False
        
        # Cancel all watcher tasks
        for watcher in self.watchers.values():
            if watcher.task and not watcher.task.done():
                watcher.task.cancel()
        
        # Cancel cleanup task
        if self.cleanup_task and not self.cleanup_task.done():
            self.cleanup_task.cancel()
        
        # Wait for tasks to complete
        tasks = [w.task for w in self.watchers.values() if w.task]
        if self.cleanup_task:
            tasks.append(self.cleanup_task)
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        
        logging.info("Stopped all watchers")
    
    async def _cleanup_loop(self):
        """Periodic cleanup of failed watchers."""
        while self.running:
            try:
                await asyncio.sleep(300)  # 5 minutes
                
                # Remove stopped/error watchers
                to_remove = []
                for symbol, watcher in self.watchers.items():
                    if watcher.status in [WatcherStatus.STOPPED, WatcherStatus.ERROR]:
                        to_remove.append(symbol)
                
                for symbol in to_remove:
                    del self.watchers[symbol]
                    logging.info(f"Cleaned up watcher for {symbol}")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.error(f"Error in cleanup loop: {e}")
    
    async def _default_callback(self, symbol: str, data: Dict):
        """
        Default callback for watcher alerts.

        Args:
            symbol: Stock symbol that triggered alert
            data: Technical analysis data

        Example Usage:
            # Called automatically when watcher condition is met
            await watcher._default_callback("AAPL", {
                "rsi": 65,
                "macd_signal": "bullish",
                "trend": "uptrend",
                "price": 185.50
            })

        Example Output:
            # Logs:
            # 🎯 ALERT: AAPL - Bullish signals detected!
            #    RSI: 65
            #    MACD: bullish
            #    Trend: uptrend
            #    Price: $185.50
        """
        logging.info(f"🎯 ALERT: {symbol} - Bullish signals detected!")
        logging.info(f"   RSI: {data.get('rsi', 'N/A')}")
        logging.info(f"   MACD: {data.get('macd_signal', 'N/A')}")
        logging.info(f"   Trend: {data.get('trend', 'N/A')}")
        logging.info(f"   Price: ${data.get('price', 'N/A')}")


# Example usage functions
def create_breakout_watcher(manager: WatcherManager, symbol: str):
    """
    Create a breakout watcher.

    Args:
        manager: WatcherManager instance
        symbol: Stock symbol to watch

    Returns:
        Watcher ID

    Example Usage:
        # Create breakout watcher for TSLA
        watcher_id = create_breakout_watcher(watcher_manager, "TSLA")

    Example Output:
        "TSLA"  # Returns the symbol as watcher ID
    """

    def breakout_condition(data):
        return (
            data.get("volume_trend") == "surge"
            and data.get("price", 0) > data.get("resistance", 0)
            and data.get("rsi", 50) > 60
        )
    
    async def breakout_callback(symbol, data):
        logging.info(f"🚀 BREAKOUT: {symbol} at ${data.get('price', 'N/A')}")
    
    return manager.register_watcher(
        symbol=symbol, interval=10.0, condition=breakout_condition, callback=breakout_callback  # 10 second polling
    )


def create_oversold_watcher(manager: WatcherManager, symbol: str):
    """
    Create an oversold bounce watcher.

    Args:
        manager: WatcherManager instance
        symbol: Stock symbol to watch

    Returns:
        Watcher ID

    Example Usage:
        # Create oversold watcher for AMD
        watcher_id = create_oversold_watcher(watcher_manager, "AMD")

    Example Output:
        "AMD"  # Returns the symbol as watcher ID
    """

    def oversold_condition(data):
        return (
            data.get("rsi", 50) < 30
            and data.get("trend") != "downtrend"
            and data.get("volume_trend") in ["increasing", "surge"]
        )
    
    async def oversold_callback(symbol, data):
        logging.info(f"📈 OVERSOLD BOUNCE: {symbol} at ${data.get('price', 'N/A')}")
    
    return manager.register_watcher(
        symbol=symbol, interval=60.0, condition=oversold_condition, callback=oversold_callback  # 1 minute polling
    )
