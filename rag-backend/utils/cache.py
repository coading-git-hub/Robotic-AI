from typing import Any, Optional
import asyncio
import time
import hashlib
from functools import wraps
from datetime import datetime, timedelta

# Simple in-memory cache implementation
# For production, consider using Redis or another caching solution
class SimpleCache:
    def __init__(self):
        self._cache = {}
        self._expirations = {}

    def set(self, key: str, value: Any, ttl: int = 300) -> None:  # Default TTL: 5 minutes
        """Set a value in cache with TTL in seconds"""
        self._cache[key] = value
        self._expirations[key] = datetime.now() + timedelta(seconds=ttl)

    def get(self, key: str) -> Optional[Any]:
        """Get a value from cache, returns None if expired or not found"""
        if key in self._expirations and datetime.now() > self._expirations[key]:
            # Remove expired entry
            del self._cache[key]
            del self._expirations[key]
            return None

        return self._cache.get(key)

    def delete(self, key: str) -> bool:
        """Delete a key from cache"""
        if key in self._cache:
            del self._cache[key]
            if key in self._expirations:
                del self._expirations[key]
            return True
        return False

    def clear(self) -> None:
        """Clear all cache entries"""
        self._cache.clear()
        self._expirations.clear()

    def cleanup_expired(self) -> int:
        """Remove all expired entries and return count of removed entries"""
        now = datetime.now()
        expired_keys = [
            key for key, exp in self._expirations.items()
            if now > exp
        ]

        for key in expired_keys:
            del self._cache[key]
            del self._expirations[key]

        return len(expired_keys)


# Global cache instance
_cache = SimpleCache()


def cached(ttl: int = 300):
    """Decorator to cache function results"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Create cache key from function name and arguments
            cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"

            # Try to get from cache
            cached_result = _cache.get(cache_key)
            if cached_result is not None:
                return cached_result

            # Execute function and cache result
            result = await func(*args, **kwargs)
            _cache.set(cache_key, result, ttl)
            return result

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Create cache key from function name and arguments
            cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"

            # Try to get from cache
            cached_result = _cache.get(cache_key)
            if cached_result is not None:
                return cached_result

            # Execute function and cache result
            result = func(*args, **kwargs)
            _cache.set(cache_key, result, ttl)
            return result

        # Return the appropriate wrapper based on whether the function is async
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def get_cache():
    """Get the global cache instance"""
    return _cache