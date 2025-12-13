from fastapi import Request, FastAPI
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from slowapi.errors import RateLimitExceeded
from pydantic import ValidationError
from typing import Dict, Any
import logging
import traceback
import uuid
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init_error_handlers(app: FastAPI):
    """Initialize error handlers for the FastAPI application"""

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        """Handle HTTP exceptions"""
        request_id = str(uuid.uuid4())
        logger.error(f"Request {request_id}: HTTP {exc.status_code} - {exc.detail}")

        return JSONResponse(
            status_code=exc.status_code,
            content={
                "success": False,
                "error": {
                    "code": f"HTTP_{exc.status_code}",
                    "message": str(exc.detail),
                    "details": {}
                },
                "timestamp": datetime.utcnow().isoformat(),
                "request_id": request_id
            }
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """Handle request validation errors"""
        request_id = str(uuid.uuid4())
        logger.error(f"Request {request_id}: Validation error - {exc}")

        return JSONResponse(
            status_code=422,
            content={
                "success": False,
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": "Request validation failed",
                    "details": {
                        "errors": [
                            {
                                "loc": error["loc"],
                                "msg": error["msg"],
                                "type": error["type"]
                            } for error in exc.errors()
                        ]
                    }
                },
                "timestamp": datetime.utcnow().isoformat(),
                "request_id": request_id
            }
        )

    @app.exception_handler(RateLimitExceeded)
    async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
        """Handle rate limit exceeded errors"""
        request_id = str(uuid.uuid4())
        logger.warning(f"Request {request_id}: Rate limit exceeded")

        return JSONResponse(
            status_code=429,
            content={
                "success": False,
                "error": {
                    "code": "RATE_LIMIT_EXCEEDED",
                    "message": "Rate limit exceeded. Please try again later.",
                    "details": {
                        "retry_after": exc.retry_after if hasattr(exc, 'retry_after') else None
                    }
                },
                "timestamp": datetime.utcnow().isoformat(),
                "request_id": request_id
            }
        )

    @app.exception_handler(ValidationError)
    async def pydantic_validation_handler(request: Request, exc: ValidationError):
        """Handle Pydantic validation errors"""
        request_id = str(uuid.uuid4())
        logger.error(f"Request {request_id}: Pydantic validation error - {exc}")

        return JSONResponse(
            status_code=422,
            content={
                "success": False,
                "error": {
                    "code": "PYDANTIC_VALIDATION_ERROR",
                    "message": "Data validation failed",
                    "details": {
                        "errors": [
                            {
                                "loc": error["loc"],
                                "msg": error["msg"],
                                "type": error["type"]
                            } for error in exc.errors()
                        ]
                    }
                },
                "timestamp": datetime.utcnow().isoformat(),
                "request_id": request_id
            }
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle general exceptions"""
        request_id = str(uuid.uuid4())
        logger.error(f"Request {request_id}: Internal server error - {exc}")
        logger.error(traceback.format_exc())

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": "An internal server error occurred",
                    "details": {
                        "request_id": request_id
                    }
                },
                "timestamp": datetime.utcnow().isoformat(),
                "request_id": request_id
            }
        )

# Create a custom exception class for application-specific errors
class AppException(Exception):
    """Base application exception class"""
    def __init__(self, message: str, code: str = "APP_ERROR", status_code: int = 500, details: Dict[str, Any] = None):
        self.message = message
        self.code = code
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)

class ContentNotFoundException(AppException):
    """Exception raised when requested content is not found"""
    def __init__(self, content_id: str):
        super().__init__(
            message=f"Content with ID {content_id} not found",
            code="CONTENT_NOT_FOUND",
            status_code=404
        )

class UserNotFoundException(AppException):
    """Exception raised when requested user is not found"""
    def __init__(self, user_id: str):
        super().__init__(
            message=f"User with ID {user_id} not found",
            code="USER_NOT_FOUND",
            status_code=404
        )

class AuthenticationException(AppException):
    """Exception raised for authentication errors"""
    def __init__(self, message: str = "Authentication failed"):
        super().__init__(
            message=message,
            code="AUTHENTICATION_ERROR",
            status_code=401
        )

class AuthorizationException(AppException):
    """Exception raised for authorization errors"""
    def __init__(self, message: str = "Insufficient permissions"):
        super().__init__(
            message=message,
            code="AUTHORIZATION_ERROR",
            status_code=403
        )