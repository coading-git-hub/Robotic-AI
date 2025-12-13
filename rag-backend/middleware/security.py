from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import time
import logging
from typing import Optional

# Try to import rate limiting, but make it optional
try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
    RATE_LIMITING_AVAILABLE = True
except ImportError:
    RATE_LIMITING_AVAILABLE = False

# Initialize rate limiter if available
if RATE_LIMITING_AVAILABLE:
    limiter = Limiter(key_func=get_remote_address)

logger = logging.getLogger(__name__)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware to add security headers to responses"""

    async def dispatch(self, request: Request, call_next):
        # Add security headers to the response
        response: Response = await call_next(request)

        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data: https:;"

        # Prevent MIME-type sniffing
        if "Content-Type" not in response.headers:
            response.headers["Content-Type"] = "application/json; charset=utf-8"

        return response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to log incoming requests"""

    async def dispatch(self, request: Request, call_next):
        start_time = time.time()

        # Log the incoming request
        logger.info(
            f"REQUEST_START: {request.method} {request.url.path} "
            f"from {request.client.host}:{request.client.port if request.client.port else 'unknown'}"
        )

        try:
            response = await call_next(request)
        except Exception as e:
            # Log the error
            logger.error(
                f"REQUEST_ERROR: {request.method} {request.url.path} "
                f"error: {str(e)}"
            )
            raise

        # Calculate response time
        response_time = time.time() - start_time

        # Log the response
        logger.info(
            f"REQUEST_END: {request.method} {request.url.path} "
            f"status={response.status_code} "
            f"response_time={response_time:.3f}s"
        )

        return response


class InputValidationMiddleware(BaseHTTPMiddleware):
    """Middleware to validate and sanitize input"""

    async def dispatch(self, request: Request, call_next):
        # For POST/PUT requests, validate the content
        if request.method in ["POST", "PUT", "PATCH"]:
            try:
                # Read the request body
                body_bytes = await request.body()
                body_str = body_bytes.decode('utf-8')

                # For now, we just pass through - in a real implementation
                # we would validate and sanitize the input here
                # using the validation functions from utils.security

                # Reset the request body for the next middleware
                # This is a simplified approach - in practice, you'd need to handle
                # this differently based on your specific validation requirements
                request._body = body_bytes

            except UnicodeDecodeError:
                return JSONResponse(
                    status_code=400,
                    content={"detail": "Invalid character encoding in request body"}
                )
            except Exception as e:
                logger.error(f"Error in input validation middleware: {e}")
                return JSONResponse(
                    status_code=500,
                    content={"detail": "Internal server error in input validation"}
                )

        response = await call_next(request)
        return response