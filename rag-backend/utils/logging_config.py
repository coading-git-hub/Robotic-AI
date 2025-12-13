import logging
import sys
import json
from datetime import datetime
from typing import Dict, Any
from enum import Enum
from pythonjsonlogger import jsonlogger
from core.config import settings


class LogType(str, Enum):
    """Log types for structured logging"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    DEBUG = "debug"
    CRITICAL = "critical"
    REQUEST = "request"
    RESPONSE = "response"
    DATABASE = "database"
    SECURITY = "security"


class CustomJsonFormatter(jsonlogger.JsonFormatter):
    """Custom JSON formatter for structured logging"""

    def add_fields(self, log_record, record, message_dict):
        super(CustomJsonFormatter, self).add_fields(log_record, record, message_dict)

        # Add custom fields
        log_record['timestamp'] = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%fZ')
        log_record['level'] = record.levelname
        log_record['service'] = 'physical-ai-rag'
        log_record['version'] = '1.0.0'

        if hasattr(record, 'request_id'):
            log_record['request_id'] = record.request_id

        if hasattr(record, 'user_id'):
            log_record['user_id'] = record.user_id


def setup_logging():
    """Set up structured logging for the application"""

    # Create formatters
    if settings.DEBUG:
        # Human-readable format for development
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    else:
        # JSON format for production
        formatter = CustomJsonFormatter(
            '%(timestamp)s %(service)s %(version)s %(level)s %(name)s %(message)s',
            rename_fields={'name': 'logger', 'levelname': 'level'}
        )

    # Create handlers
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG if settings.DEBUG else logging.INFO)
    root_logger.addHandler(handler)

    # Configure specific loggers
    uvicorn_logger = logging.getLogger("uvicorn")
    uvicorn_logger.setLevel(logging.INFO)

    fastapi_logger = logging.getLogger("fastapi")
    fastapi_logger.setLevel(logging.INFO)

    # SQLAlchemy logger for database queries (only in debug mode)
    if settings.DEBUG:
        sqlalchemy_logger = logging.getLogger("sqlalchemy.engine")
        sqlalchemy_logger.setLevel(logging.INFO)


def get_logger(name: str) -> logging.Logger:
    """Get a configured logger instance"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG if settings.DEBUG else logging.INFO)
    return logger


def log_request(request_id: str, method: str, url: str, user_id: str = None):
    """Log an incoming request"""
    logger = get_logger("request")
    extra = {'request_id': request_id}
    if user_id:
        extra['user_id'] = user_id

    logger.info(
        "Incoming request",
        extra=extra,
        extra_data={
            "method": method,
            "url": url,
            "type": LogType.REQUEST
        }
    )


def log_response(request_id: str, status_code: int, response_time: float, user_id: str = None):
    """Log an outgoing response"""
    logger = get_logger("response")
    extra = {'request_id': request_id}
    if user_id:
        extra['user_id'] = user_id

    logger.info(
        "Outgoing response",
        extra=extra,
        extra_data={
            "status_code": status_code,
            "response_time_ms": response_time,
            "type": LogType.RESPONSE
        }
    )


def log_database_query(request_id: str, query: str, execution_time: float, user_id: str = None):
    """Log a database query"""
    logger = get_logger("database")
    extra = {'request_id': request_id}
    if user_id:
        extra['user_id'] = user_id

    logger.info(
        "Database query executed",
        extra=extra,
        extra_data={
            "query": query,
            "execution_time_ms": execution_time,
            "type": LogType.DATABASE
        }
    )


def log_security_event(event_type: str, request_id: str, user_id: str = None, details: Dict[str, Any] = None):
    """Log a security-related event"""
    logger = get_logger("security")
    extra = {'request_id': request_id}
    if user_id:
        extra['user_id'] = user_id

    logger.warning(
        f"Security event: {event_type}",
        extra=extra,
        extra_data={
            "event_type": event_type,
            "details": details or {},
            "type": LogType.SECURITY
        }
    )


def log_error(request_id: str, error: Exception, user_id: str = None, context: Dict[str, Any] = None):
    """Log an error with context"""
    logger = get_logger("error")
    extra = {'request_id': request_id}
    if user_id:
        extra['user_id'] = user_id

    logger.error(
        str(error),
        extra=extra,
        extra_data={
            "error_type": type(error).__name__,
            "context": context or {},
            "type": LogType.ERROR
        }
    )


# Initialize logging when module is imported
setup_logging()