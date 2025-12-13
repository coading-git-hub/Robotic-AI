from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi import FastAPI, Request
from core.config import settings
import time


# Initialize the rate limiter with default limits
def init_rate_limiter(app: FastAPI):
    """Initialize rate limiting for the application"""
    # Create limiter with default storage and key function
    limiter = Limiter(
        key_func=get_remote_address,  # Use IP address as the key
        default_limits=[f"{settings.RATE_LIMIT_REQUESTS} per {settings.RATE_LIMIT_WINDOW} seconds"]
    )

    # Attach the limiter to the app
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    return limiter


# Specific rate limit decorators for different endpoints
from slowapi import Limiter
from functools import wraps

def get_limiter():
    """Get the limiter instance"""
    # This will be initialized when the app starts
    from main import app  # Import app to get its state
    return app.state.limiter


def auth_rate_limit():
    """Rate limit for authentication endpoints"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            limiter = get_limiter()
            # Apply a more restrictive rate limit for auth endpoints
            # This prevents brute force attacks
            return await limiter.limit("5/minute")(func)(*args, **kwargs)
        return wrapper
    return decorator


def query_rate_limit():
    """Rate limit for query endpoints"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            limiter = get_limiter()
            # Apply rate limit for query endpoints to prevent abuse
            return await limiter.limit("10/minute")(func)(*args, **kwargs)
        return wrapper
    return decorator


def general_rate_limit():
    """General rate limit for other endpoints"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            limiter = get_limiter()
            # Apply default rate limit
            return await limiter.limit(f"{settings.RATE_LIMIT_REQUESTS} per {settings.RATE_LIMIT_WINDOW} seconds")(func)(*args, **kwargs)
        return wrapper
    return decorator


# Example usage in endpoints:
#
# @router.post("/query")
# @query_rate_limit()
# async def query_endpoint(request: Request, query: QueryRequest):
#     # Your endpoint logic here
#     pass