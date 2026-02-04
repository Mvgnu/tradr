import json
import os
import logging
import datetime as dt
from typing import Dict, List, Any, Optional
from filelock import FileLock
from tradr.core.models import MarketCondition, TradeData, PerformanceMetrics, safe_serialize

class AgentMemory:
    """Manages agent memory with efficient watchlist pruning. Watchlist is always a list of dicts: {symbol, added_at, reason}. Legacy string entries are migrated on load."""
    
    def __init__(self, file_path: str = "agent_memory.json"):
        self.file_path = file_path
        self.memory = self._load_memory()
        # Migrate legacy watchlist entries to dicts
        self._migrate_watchlist()
    
    def _load_memory(self) -> Dict[str, Any]:
        logging.debug(f"[MEMORY] _load_memory START {self.file_path}")
        lock_path = self.file_path + ".lock"
        logging.debug(f"[MEMORY] _load_memory waiting for lock {lock_path}")
        with FileLock(lock_path, timeout=10):
            logging.debug(f"[MEMORY] _load_memory acquired lock {lock_path}")
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, 'r') as f:
                        data = json.load(f)
                        logging.debug(f"[MEMORY] _load_memory loaded data: {type(data)}")
                        logging.debug(f"[MEMORY] _load_memory END {self.file_path}")
                        return data
            except (json.JSONDecodeError, IOError):
                logging.debug(f"[MEMORY] _load_memory failed to load, using default")
                logging.debug(f"[MEMORY] _load_memory END {self.file_path}")
                return self._default_memory()
        logging.debug(f"[MEMORY] _load_memory file not found, using default")
        logging.debug(f"[MEMORY] _load_memory END {self.file_path}")
        return self._default_memory()
    
    def _default_memory(self) -> Dict[str, Any]:
        """Default memory structure."""
        return {
            "watchlist": [],
            "market_conditions": [],
            "trades": [],
            "performance_history": [],
            "daily_snapshots": [], # NEW: For TWRR
            "performance_metrics": {
                "total_trades": 0,
                "winning_trades": 0,
                "total_pnl": 0.0,
                "max_drawdown": 0.0,
                "sharpe_ratio": 0.0
            },
            "learning_insights": [],
            "error_count": 0,
            "safe_mode": False
        }
    
    def save_memory(self):
        logging.debug(f"[MEMORY] save_memory START {self.file_path}")
        lock_path = self.file_path + ".lock"
        try:
            logging.debug(f"[MEMORY] save_memory waiting for lock {lock_path}")
            with FileLock(lock_path, timeout=10):
                logging.debug(f"[MEMORY] save_memory acquired lock {lock_path}")
            with open(self.file_path, 'w') as f:
                json.dump(self.memory, f, indent=2)
                logging.debug(f"[MEMORY] save_memory END {self.file_path}")
        except IOError as e:
            print(f"Error saving memory: {e}")
    
    def _migrate_watchlist(self):
        """Convert any legacy string watchlist entries to dicts."""
        current_time = dt.datetime.now(dt.timezone.utc)
        wl = self.memory.get("watchlist", [])
        new_wl = []
        for item in wl:
            if isinstance(item, dict) and "symbol" in item:
                new_wl.append(item)
            else:
                new_wl.append({
                    "symbol": item if isinstance(item, str) else str(item),
                    "added_at": current_time.isoformat(),
                    "reason": "legacy_item"
                })
        self.memory["watchlist"] = new_wl

    def get_watchlist(self) -> List[Dict]:
        logging.debug(f"[MEMORY] get_watchlist START")
        wl = self.memory.get("watchlist", []).copy()
        logging.debug(f"[MEMORY] get_watchlist END (len={len(wl)})")
        return wl
    
    def prune_watchlist(self):
        """Explicit watchlist pruning to keep getter pure."""
        current_time = dt.datetime.now(dt.timezone.utc)
        
        pruned_watchlist = []
        for item in self.memory.get("watchlist", []):
            if isinstance(item, dict) and "added_at" in item:
                try:
                    added_at = dt.datetime.fromisoformat(item["added_at"])
                    if (current_time - added_at).total_seconds() < ttl_hours * 3600:
                        pruned_watchlist.append(item)
                except (ValueError, TypeError):
                    # Keep items with invalid timestamps for manual review
                    pruned_watchlist.append(item)
            else:
                # Handle legacy string format
                pruned_watchlist.append({
                    "symbol": item if isinstance(item, str) else str(item),
                    "added_at": current_time.isoformat(),
                    "reason": "legacy_item"
                })
        
        self.memory["watchlist"] = pruned_watchlist
    
    def add_to_watchlist(self, symbol: str, reason: str = ""):
        logging.debug(f"[MEMORY] add_to_watchlist START symbol={symbol}, reason={reason}")
        current_time = dt.datetime.now(dt.timezone.utc)
        
        # Check if symbol already exists
        for item in self.memory.get("watchlist", []):
            symbol_key = item.get("symbol") if isinstance(item, dict) else item
            if symbol_key == symbol:
                logging.debug(f"[MEMORY] add_to_watchlist END (already exists) symbol={symbol}")
                return  # Already in watchlist
        
        watchlist_item = {
            "symbol": symbol,
            "added_at": current_time.isoformat(),
            "reason": reason
        }
        
        self.memory.setdefault("watchlist", []).append(watchlist_item)
        logging.debug(f"[MEMORY] add_to_watchlist END symbol={symbol}")
    
    def remove_from_watchlist(self, symbol: str):
        logging.debug(f"[MEMORY] remove_from_watchlist START symbol={symbol}")
        watchlist = self.memory.get("watchlist", [])
        
        updated_watchlist = []
        for item in watchlist:
            symbol_key = item.get("symbol") if isinstance(item, dict) else item
            if symbol_key != symbol:
                updated_watchlist.append(item)
        
        self.memory["watchlist"] = updated_watchlist
        logging.debug(f"[MEMORY] remove_from_watchlist END symbol={symbol}")
    
    def record_market_condition(self, condition: MarketCondition):
        """Record market condition with rotation."""
        conditions = self.memory.setdefault("market_conditions", [])
        conditions.append(condition.model_dump(mode='json'))
        
        # Keep only last 100 conditions
        if len(conditions) > 100:
            self.memory["market_conditions"] = conditions[-100:]
    
    def update_performance_metrics(self, metrics: Dict[str, Any]):
        """Update performance metrics."""
        current_metrics = self.memory.setdefault("performance_metrics", {})
        current_metrics.update(metrics)
    
    def add_learning_insight(self, insight: str):
        """Add learning insight with rotation."""
        insights = self.memory.setdefault("learning_insights", [])
        insights.append({
            "insight": insight,
            "timestamp": dt.datetime.now(dt.timezone.utc).isoformat()
        })
        
        # Keep only last 50 insights
        if len(insights) > 50:
            self.memory["learning_insights"] = insights[-50:]
    
    def increment_error_count(self):
        """Increment error count and check for safe mode."""
        self.memory["error_count"] = self.memory.get("error_count", 0) + 1
        
        if self.memory["error_count"] >= 3:
            self.memory["safe_mode"] = True
    
    def reset_error_count(self):
        """Reset error count and exit safe mode."""
        self.memory["error_count"] = 0
        self.memory["safe_mode"] = False
    
    def is_safe_mode(self) -> bool:
        """Check if agent is in safe mode."""
        return self.memory.get("safe_mode", False)
    
    def get_recent_insights(self, limit: int = 10) -> List[Dict]:
        """Get recent learning insights."""
        insights = self.memory.get("learning_insights", [])
        return insights[-limit:] if insights else []

    def save_notes(self, notes: str) -> Dict:
        logging.debug(f"[MEMORY] save_notes START notes={notes}")
        self.memory["agent_notes"] = notes
        self.save_memory()
        logging.debug(f"[MEMORY] save_notes END notes={notes}")
        return {"status": "success"}

    def log_trade(self, trade_data: Dict) -> Dict:
        """Log trade for performance tracking using Pydantic model."""
        try:
            # Ensure timestamp is set
            if 'timestamp' not in trade_data:
                trade_data['timestamp'] = dt.datetime.now(dt.timezone.utc).isoformat()
            
            # Validate and serialize using Pydantic model
            trade = TradeData(**trade_data)
            self.memory.setdefault('trades', []).append(trade.model_dump(mode='json'))
            self.save_memory()
            return {"status": "success"}
        except Exception as e:
            logging.error(f"Error logging trade: {e}")
            return {"status": "error", "message": str(e)}

    def get_last_trade_for_symbol(self, symbol: str) -> Optional[Dict]:
        """Get most recent trade for a symbol."""
        trades = self.memory.get("trades", [])
        if not trades:
            return None
        symbol_upper = symbol.upper()
        for trade in reversed(trades):
            try:
                trade_symbol = str(trade.get("symbol", "")).upper()
            except Exception:
                trade_symbol = ""
            if trade_symbol == symbol_upper:
                return trade
        return None

    def update_performance(self, metrics: Dict) -> Dict:
        """Track performance metrics over time using Pydantic model."""
        try:
            # Ensure timestamp is set
            if 'timestamp' not in metrics:
                metrics['timestamp'] = dt.datetime.now(dt.timezone.utc).isoformat()
            
            # Use safe_serialize for complex objects
            clean_metrics = safe_serialize(metrics)
            self.memory.setdefault('performance_history', []).append(clean_metrics)
            
            # Keep only last 100 entries
            if len(self.memory['performance_history']) > 100:
                self.memory['performance_history'] = self.memory['performance_history'][-100:]
            
            self.save_memory()
            return {"status": "success"}
        except Exception as e:
            logging.error(f"Error updating performance: {e}")
            return {"status": "error", "message": str(e)}

    def get_strategy_parameters(self) -> Dict:
        """Get adaptive strategy parameters."""
        return self.memory.get('strategy_parameters', {
            'min_rsi_oversold': 30,
            'max_rsi_overbought': 70,
            'trend_confirmation_days': 3,
            'stop_loss_atr_multiple': 1.5,
            'take_profit_atr_multiple': 2.5
        })

    def get_notes(self) -> str:
        """Get current agent notes."""
        return self.memory.get("agent_notes", "")

    def log_daily_snapshot(self, market_value: float):
        """Logs the total market value of the portfolio at a point in time."""
        snapshots = self.memory.setdefault("daily_snapshots", [])
        snapshot_data = {
            "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
            "market_value": market_value
        }
        # Avoid duplicate snapshots for the same day
        today_str = dt.datetime.now(dt.timezone.utc).date().isoformat()
        if not any(s['timestamp'].startswith(today_str) for s in snapshots):
            snapshots.append(snapshot_data)
            # Keep a reasonable history, e.g., last 3 years
            if len(snapshots) > 365 * 3:
                self.memory["daily_snapshots"] = snapshots[-(365*3):]
            self.save_memory()
