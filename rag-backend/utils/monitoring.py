from typing import Dict, Any, Optional
import time
import logging
from datetime import datetime
from functools import wraps
import asyncio
from enum import Enum

# Configure logger for monitoring
monitoring_logger = logging.getLogger("monitoring")
monitoring_logger.setLevel(logging.INFO)


class EventType(Enum):
    """Types of events that can be monitored"""
    API_CALL = "api_call"
    DB_OPERATION = "db_operation"
    RAG_QUERY = "rag_query"
    USER_ACTION = "user_action"
    SYSTEM_EVENT = "system_event"
    ERROR = "error"


class MonitoringService:
    """Service for monitoring application performance and events"""

    def __init__(self):
        self.metrics = {
            "api_calls": 0,
            "errors": 0,
            "total_response_time": 0.0,
            "avg_response_time": 0.0,
        }
        self.start_time = datetime.utcnow()

    def log_event(self, event_type: EventType, details: Dict[str, Any], user_id: Optional[str] = None):
        """Log an event with details"""
        event_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type.value,
            "details": details,
            "user_id": user_id
        }

        monitoring_logger.info(f"MONITORING_EVENT: {event_data}")

    def track_api_call(self, endpoint: str, method: str, response_time: float, user_id: Optional[str] = None):
        """Track an API call for monitoring"""
        self.metrics["api_calls"] += 1
        self.metrics["total_response_time"] += response_time

        if self.metrics["api_calls"] > 0:
            self.metrics["avg_response_time"] = (
                self.metrics["total_response_time"] / self.metrics["api_calls"]
            )

        self.log_event(
            EventType.API_CALL,
            {
                "endpoint": endpoint,
                "method": method,
                "response_time": response_time,
                "avg_response_time": self.metrics["avg_response_time"]
            },
            user_id
        )

    def track_error(self, error_type: str, error_message: str, endpoint: Optional[str] = None, user_id: Optional[str] = None):
        """Track an error for monitoring"""
        self.metrics["errors"] += 1

        self.log_event(
            EventType.ERROR,
            {
                "error_type": error_type,
                "error_message": error_message,
                "endpoint": endpoint
            },
            user_id
        )

    def get_system_uptime(self) -> float:
        """Get system uptime in seconds"""
        return (datetime.utcnow() - self.start_time).total_seconds()

    def get_metrics(self) -> Dict[str, Any]:
        """Get current system metrics"""
        return {
            **self.metrics,
            "uptime_seconds": self.get_system_uptime(),
            "timestamp": datetime.utcnow().isoformat()
        }

    def api_monitor(self, endpoint_name: str):
        """Decorator to monitor API endpoints"""
        def decorator(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                start_time = time.time()
                user_id = None  # Extract user_id from request context if available

                try:
                    result = await func(*args, **kwargs)
                    response_time = time.time() - start_time
                    self.track_api_call(endpoint_name, "POST/GET/PUT/DELETE", response_time, user_id)
                    return result
                except Exception as e:
                    response_time = time.time() - start_time
                    self.track_error(type(e).__name__, str(e), endpoint_name, user_id)
                    raise

            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                start_time = time.time()
                user_id = None  # Extract user_id from request context if available

                try:
                    result = func(*args, **kwargs)
                    response_time = time.time() - start_time
                    self.track_api_call(endpoint_name, "POST/GET/PUT/DELETE", response_time, user_id)
                    return result
                except Exception as e:
                    response_time = time.time() - start_time
                    self.track_error(type(e).__name__, str(e), endpoint_name, user_id)
                    raise

            # Return the appropriate wrapper based on whether the function is async
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper

        return decorator


# Global monitoring instance
_monitoring_service = MonitoringService()


def get_monitoring_service() -> MonitoringService:
    """Get the global monitoring service instance"""
    return _monitoring_service


def monitor_api_call(endpoint_name: str):
    """Decorator to monitor API calls"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            user_id = None  # Would extract from request context in real implementation

            try:
                result = await func(*args, **kwargs)
                response_time = time.time() - start_time
                _monitoring_service.track_api_call(endpoint_name, "API", response_time, user_id)
                return result
            except Exception as e:
                response_time = time.time() - start_time
                _monitoring_service.track_error(type(e).__name__, str(e), endpoint_name, user_id)
                raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            user_id = None  # Would extract from request context in real implementation

            try:
                result = func(*args, **kwargs)
                response_time = time.time() - start_time
                _monitoring_service.track_api_call(endpoint_name, "API", response_time, user_id)
                return result
            except Exception as e:
                response_time = time.time() - start_time
                _monitoring_service.track_error(type(e).__name__, str(e), endpoint_name, user_id)
                raise

        # Return the appropriate wrapper based on whether the function is async
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator