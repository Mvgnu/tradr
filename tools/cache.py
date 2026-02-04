import sqlite3
import json
import time
import hashlib
import os
from typing import Any, Optional, Callable
from functools import wraps


class SQLiteCache:
    """SQLite-based cache with TTL for API responses."""

    def __init__(self, db_path: str = "/tmp/tradr_cache.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize the cache database."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    timestamp INTEGER,
                    ttl INTEGER
                )
            """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON cache(timestamp)")

    def _make_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Create a cache key from function name and arguments."""
        key_data = {"func": func_name, "args": args, "kwargs": kwargs}
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_str.encode()).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT value, timestamp, ttl FROM cache WHERE key = ?", (key,))
            row = cursor.fetchone()

            if row is None:
                return None

            value, timestamp, ttl = row

            if time.time() - timestamp > ttl:
                conn.execute("DELETE FROM cache WHERE key = ?", (key,))
                return None

            return json.loads(value)

    def set(self, key: str, value: Any, ttl: int = 600):
        """Set value in cache with TTL."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO cache (key, value, timestamp, ttl) VALUES (?, ?, ?, ?)",
                (key, json.dumps(value, default=str), time.time(), ttl),
            )

    def cleanup(self):
        """Remove expired entries."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM cache WHERE timestamp + ttl < ?", (time.time(),))


# Global cache instance
_cache = SQLiteCache()


def cache(ttl: int = 600):
    """Decorator to cache function results."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key
            cache_key = _cache._make_key(func.__name__, args, kwargs)

            # Try to get from cache
            cached_result = _cache.get(cache_key)
            if cached_result is not None:
                return cached_result

            # Execute function and cache result
            result = func(*args, **kwargs)
            _cache.set(cache_key, result, ttl)

            return result

        return wrapper

    return decorator


def clear_cache():
    """Clear all cache entries."""
    with sqlite3.connect(_cache.db_path) as conn:
        conn.execute("DELETE FROM cache")


def cleanup_cache():
    """Remove expired cache entries."""
    _cache.cleanup()
