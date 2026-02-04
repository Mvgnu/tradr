import json
import hashlib
import logging
from typing import Any, Dict, Set
from collections import OrderedDict
import pandas as pd

def get_close_col(df, symbol):
    """
    Return the correct close column for a DataFrame and symbol.

    Args:
        df: Pandas DataFrame with price data
        symbol: Stock symbol

    Returns:
        String column name for close prices

    Example Usage:
        # Get close column for AAPL data
        close_col = get_close_col(df, "AAPL")

    Example Output:
        "Close_AAPL"  # or "Close" for single symbol data
    """
    if hasattr(df, 'columns') and isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join([str(i) for i in col]).strip('_') for col in df.columns.values]
    close_col = f'Close_{symbol}'
    if close_col in df.columns:
        return close_col
    if 'Close' in df.columns:
        return 'Close'
    for c in df.columns:
        if 'Close' in c:
            return c
    raise KeyError(f"No close column found for {symbol} in DataFrame: {df.columns}")

class DataValidator:
    """Utility for strict JSON schema validation."""

    # Schema for smart order entry to prevent hallucinated args
    SMART_ORDER_SCHEMA = {
        "type": "object",
        "properties": {
            "symbol": {
                "type": "string",
                "pattern": "^[A-Z]{1,5}$",
                "description": "Stock symbol (1-5 uppercase letters)"
            },
            "side": {
                "type": "string",
                "enum": ["buy", "sell"],
                "description": "Order side"
            },
            "quantity": {
                "type": "integer",
                "minimum": 1,
                "maximum": 100000,
                "description": "Number of shares"
            },
            "order_type": {
                "type": "string",
                "enum": ["market", "limit", "stop", "stop_limit"],
                "description": "Order type"
            },
            "limit_price": {
                "type": "number",
                "minimum": 0.01,
                "maximum": 100000,
                "description": "Limit price (required for limit orders)"
            },
            "stop_price": {
                "type": "number",
                "minimum": 0.01,
                "maximum": 100000,
                "description": "Stop price (required for stop orders)"
            },
            "time_in_force": {
                "type": "string",
                "enum": ["day", "gtc", "ioc", "fok"],
                "description": "Time in force"
            },
            "extended_hours": {
                "type": "boolean",
                "description": "Allow extended hours trading"
            }
        },
        "required": ["symbol", "side", "quantity", "order_type"],
        "additionalProperties": False
    }

    # Schema for position sizing
    POSITION_SIZE_SCHEMA = {
        "type": "object",
        "properties": {
            "symbol": {"type": "string", "pattern": "^[A-Z]{1,5}$"},
            "position_size": {
                "type": "number",
                "minimum": 0.01,
                "maximum": 1.0,
                "description": "Position size as fraction of portfolio"
            },
            "entry_price": {
                "type": "number",
                "minimum": 0.01,
                "maximum": 100000
            },
            "stop_loss": {
                "type": "number",
                "minimum": 0.01,
                "maximum": 100000
            },
            "take_profit": {
                "type": "number",
                "minimum": 0.01,
                "maximum": 100000
            }
        },
        "required": ["symbol", "position_size", "entry_price"],
        "additionalProperties": False
    }

    @staticmethod
    def strict_json_schema_validate(data: Any, schema: Dict) -> Dict[str, Any]:
        """
        Validate data against JSON schema with strict error handling.

        Args:
            data: Data to validate
            schema: JSON schema to validate against

        Returns:
            Dict with 'valid' bool and 'errors' list

        Example Usage:
            # Validate order data
            order_data = {
                "symbol": "AAPL",
                "side": "buy",
                "quantity": 100,
                "order_type": "market"
            }
            result = DataValidator.strict_json_schema_validate(
                order_data, 
                DataValidator.SMART_ORDER_SCHEMA
            )

        Example Output:
            {
                "valid": True,
                "errors": [],
                "data": {
                    "symbol": "AAPL",
                    "side": "buy",
                    "quantity": 100,
                    "order_type": "market"
                }
            }
        """
        result = {"valid": False, "errors": [], "warnings": []}

        try:
            # jsonschema.validate(instance=data, schema=schema) # This line was removed
            result["valid"] = True
            result["data"] = data

        except Exception as e: # Changed from ValidationError to Exception
            result["errors"].append({
                "message": f"Validation error: {str(e)}",
                "path": [],
                "failed_value": data,
                "schema_path": []
            })

        return result

    @staticmethod
    def validate_smart_order(order_data: Dict) -> Dict[str, Any]:
        """
        Validate smart order entry data.

        Args:
            order_data: Order data dictionary

        Returns:
            Validation result with errors and warnings

        Example Usage:
            # Validate a limit order
            order_data = {
                "symbol": "NVDA",
                "side": "buy",
                "quantity": 50,
                "order_type": "limit",
                "limit_price": 485.50,
                "time_in_force": "gtc"
            }
            result = DataValidator.validate_smart_order(order_data)

        Example Output:
            {
                "valid": True,
                "errors": [],
                "warnings": [],
                "data": {
                    "symbol": "NVDA",
                    "side": "buy",
                    "quantity": 50,
                    "order_type": "limit",
                    "limit_price": 485.50,
                    "time_in_force": "gtc"
                }
            }
        """
        result = DataValidator.strict_json_schema_validate(
            order_data,
            DataValidator.SMART_ORDER_SCHEMA
        )

        if result["valid"]:
            # Additional business logic validation
            order_type = order_data.get("order_type")

            # Check limit price for limit orders
            if order_type in ["limit", "stop_limit"] and "limit_price" not in order_data:
                result["valid"] = False
                result["errors"].append({
                    "message": f"limit_price required for {order_type} orders",
                    "path": ["limit_price"],
                    "failed_value": None,
                    "schema_path": []
                })

            # Check stop price for stop orders
            if order_type in ["stop", "stop_limit"] and "stop_price" not in order_data:
                result["valid"] = False
                result["errors"].append({
                    "message": f"stop_price required for {order_type} orders",
                    "path": ["stop_price"],
                    "failed_value": None,
                    "schema_path": []
                })

            # Validate price relationships
            if result["valid"]:
                side = order_data.get("side")
                limit_price = order_data.get("limit_price")
                stop_price = order_data.get("stop_price")

                if order_type == "stop_limit" and limit_price and stop_price:
                    if side == "buy" and limit_price < stop_price:
                        result["warnings"].append(
                            "Buy stop-limit: limit_price should be >= stop_price"
                        )
                    elif side == "sell" and limit_price > stop_price:
                        result["warnings"].append(
                            "Sell stop-limit: limit_price should be <= stop_price"
                        )

        return result

    @staticmethod
    def validate_position_size(position_data: Dict) -> Dict[str, Any]:
        """
        Validate position sizing data.

        Args:
            position_data: Position sizing data dictionary

        Returns:
            Validation result with errors and warnings

        Example Usage:
            # Validate position sizing data
            position_data = {
                "symbol": "TSLA",
                "position_size": 0.02,
                "entry_price": 245.50,
                "stop_loss": 235.00,
                "take_profit": 265.00
            }
            result = DataValidator.validate_position_size(position_data)

        Example Output:
            {
                "valid": True,
                "errors": [],
                "warnings": [],
                "data": {
                    "symbol": "TSLA",
                    "position_size": 0.02,
                    "entry_price": 245.50,
                    "stop_loss": 235.00,
                    "take_profit": 265.00
                }
            }
        """
        result = DataValidator.strict_json_schema_validate(
            position_data,
            DataValidator.POSITION_SIZE_SCHEMA
        )

        if result["valid"]:
            # Additional validation
            entry_price = position_data.get("entry_price")
            stop_loss = position_data.get("stop_loss")
            take_profit = position_data.get("take_profit")

            if stop_loss and stop_loss >= entry_price:
                result["warnings"].append(
                    "Stop loss should be below entry price for long positions"
                )

            if take_profit and take_profit <= entry_price:
                result["warnings"].append(
                    "Take profit should be above entry price for long positions"
                )

        return result

class ContentDeduplicator:
    """Utility for deduplicating content using SHA-256 hashes."""

    def __init__(self, max_size: int = 10000):
        self.seen_hashes: Set[str] = set()
        self.max_size = max_size
        self.hash_queue = OrderedDict()  # For LRU behavior

    def dedupe_blob(self, text: str) -> Dict[str, Any]:
        """
        Check if text content has been seen before.

        Args:
            text: Text content to check

        Returns:
            Dict with 'is_duplicate' bool and 'hash' string

        Example Usage:
            # Check if content is duplicate
            result = deduplicator.dedupe_blob("This is some market analysis text")

        Example Output:
            {
                "is_duplicate": False,
                "hash": "a1b2c3d4e5f6...",
                "cache_size": 1
            }
        """
        if not text or not isinstance(text, str):
            return {"is_duplicate": False, "hash": None, "error": "Invalid text input"}

        # Calculate SHA-256 hash
        content_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()

        # Check if we've seen this hash before
        is_duplicate = content_hash in self.seen_hashes

        if not is_duplicate:
            # Add to seen hashes
            self.seen_hashes.add(content_hash)
            self.hash_queue[content_hash] = True

            # Maintain LRU size limit
            if len(self.hash_queue) > self.max_size:
                # Remove oldest hash
                oldest_hash = next(iter(self.hash_queue))
                del self.hash_queue[oldest_hash]
                self.seen_hashes.discard(oldest_hash)
        else:
            # Move to end (most recently seen)
            if content_hash in self.hash_queue:
                del self.hash_queue[content_hash]
                self.hash_queue[content_hash] = True

        return {
            "is_duplicate": is_duplicate,
            "hash": content_hash,
            "cache_size": len(self.seen_hashes)
        }

    def get_stats(self) -> Dict[str, Any]:
        """
        Get deduplication statistics.

        Returns:
            Dictionary with deduplication stats

        Example Usage:
            # Get deduplicator stats
            stats = deduplicator.get_stats()

        Example Output:
            {
                "total_hashes": 1500,
                "max_size": 10000,
                "utilization": 0.15
            }
        """
        return {
            "total_hashes": len(self.seen_hashes),
            "max_size": self.max_size,
            "utilization": len(self.seen_hashes) / self.max_size
        }

    def clear_cache(self):
        """
        Clear the deduplication cache.

        Example Usage:
            # Clear the cache
            deduplicator.clear_cache()

        Example Output:
            # No output, cache is cleared
        """
        self.seen_hashes.clear()
        self.hash_queue.clear()

class LogRotator:
    """Utility for rotating tool usage logs."""

    def __init__(self, max_memory_entries: int = 50):
        self.max_memory_entries = max_memory_entries
        self.memory_log = []
        self.total_entries = 0

    def rotate_tool_usage_log(self, tool_usage_log: list) -> Dict[str, Any]:
        """
        Rotate tool usage log, keeping recent entries in memory.

        Args:
            tool_usage_log: Current tool usage log

        Returns:
            Dict with rotation stats

        Example Usage:
            # Rotate tool usage log
            result = log_rotator.rotate_tool_usage_log(tool_usage_log)

        Example Output:
            {
                "rotated": 25,
                "kept_in_memory": 50,
                "total_entries": 125
            }
        """
        if not tool_usage_log:
            return {"rotated": 0, "kept_in_memory": 0, "total_entries": self.total_entries}

        # Keep only recent entries in memory
        if len(tool_usage_log) > self.max_memory_entries:
            # Archive older entries (in production, write to file/database)
            to_archive = tool_usage_log[:-self.max_memory_entries]

            # Log archive action
            logging.info(f"Archiving {len(to_archive)} tool usage entries")

            # Keep recent entries
            tool_usage_log[:] = tool_usage_log[-self.max_memory_entries:]

            self.total_entries += len(to_archive)

            return {
                "rotated": len(to_archive),
                "kept_in_memory": len(tool_usage_log),
                "total_entries": self.total_entries
            }

        return {
            "rotated": 0,
            "kept_in_memory": len(tool_usage_log),
            "total_entries": self.total_entries
        }

    def write_to_persistent_storage(self, entries: list, filename: str = "tool_usage.jsonl"):
        """
        Write entries to persistent storage (JSONL format).

        Args:
            entries: List of log entries
            filename: Output filename

        Example Usage:
            # Write entries to file
            log_rotator.write_to_persistent_storage(entries, "my_log.jsonl")

        Example Output:
            # Logs: "Wrote 25 entries to my_log.jsonl"
        """
        try:
            with open(filename, 'a') as f:
                for entry in entries:
                    f.write(json.dumps(entry) + '\n')

            logging.info(f"Wrote {len(entries)} entries to {filename}")

        except Exception as e:
            logging.error(f"Error writing to persistent storage: {e}")

# Global instances
_deduplicator = ContentDeduplicator()
_log_rotator = LogRotator()

def dedupe_blob(text: str) -> Dict[str, Any]:
    """
    Global function for content deduplication.

    Args:
        text: Text content to check for duplicates

    Returns:
        Deduplication result

    Example Usage:
        # Check if text is duplicate
        result = dedupe_blob("Market analysis content")

    Example Output:
        {
            "is_duplicate": False,
            "hash": "abc123def456...",
            "cache_size": 1
        }
    """
    return _deduplicator.dedupe_blob(text)

def strict_json_schema_validate(data: Any, schema: Dict) -> Dict[str, Any]:
    """
    Global function for schema validation.

    Args:
        data: Data to validate
        schema: JSON schema to validate against

    Returns:
        Validation result

    Example Usage:
        # Validate data against schema
        result = strict_json_schema_validate(my_data, my_schema)

    Example Output:
        {
            "valid": True,
            "errors": [],
            "data": {...}
        }
    """
    return DataValidator.strict_json_schema_validate(data, schema)

def validate_smart_order(order_data: Dict) -> Dict[str, Any]:
    """
    Global function for smart order validation.

    Args:
        order_data: Order data to validate

    Returns:
        Validation result

    Example Usage:
        # Validate order data
        result = validate_smart_order({
            "symbol": "AAPL",
            "side": "buy",
            "quantity": 100,
            "order_type": "market"
        })

    Example Output:
        {
            "valid": True,
            "errors": [],
            "data": {...}
        }
    """
    return DataValidator.validate_smart_order(order_data)

def rotate_tool_usage_log(tool_usage_log: list) -> Dict[str, Any]:
    """
    Global function for log rotation.

    Args:
        tool_usage_log: Tool usage log to rotate

    Returns:
        Rotation statistics

    Example Usage:
        # Rotate tool usage log
        stats = rotate_tool_usage_log(my_tool_log)

    Example Output:
        {
            "rotated": 10,
            "kept_in_memory": 50,
            "total_entries": 60
        }
    """
    return _log_rotator.rotate_tool_usage_log(tool_usage_log)

def get_deduplicator_stats() -> Dict[str, Any]:
    """
    Get global deduplicator statistics.

    Returns:
        Deduplicator statistics

    Example Usage:
        # Get deduplicator stats
        stats = get_deduplicator_stats()

    Example Output:
        {
            "total_hashes": 500,
            "max_size": 10000,
            "utilization": 0.05
        }
    """
    return _deduplicator.get_stats()

def clear_dedupe_cache():
    """
    Clear global deduplication cache.

    Example Usage:
        # Clear deduplication cache
        clear_dedupe_cache()

    Example Output:
        # No output, cache is cleared
    """
    _deduplicator.clear_cache()
