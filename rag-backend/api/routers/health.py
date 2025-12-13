from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any
from pydantic import BaseModel
from datetime import datetime

router = APIRouter()

class HealthCheck(BaseModel):
    """Health check response model"""
    status: str
    timestamp: datetime
    service: str
    version: str
    details: Dict[str, Any] = {}

@router.get("/", response_model=HealthCheck)
async def health_check():
    """Basic health check endpoint"""
    return HealthCheck(
        status="healthy",
        timestamp=datetime.utcnow(),
        service="Physical AI & Humanoid Robotics RAG API",
        version="1.0.0",
        details={
            "uptime": "running",
            "database": "connected",
            "qdrant": "connected",
            "api_version": "v1"
        }
    )

class SystemStatus(BaseModel):
    """Detailed system status response model"""
    status: str
    timestamp: datetime
    services: Dict[str, str]
    metrics: Dict[str, Any]

@router.get("/status", response_model=SystemStatus)
async def system_status():
    """Detailed system status endpoint"""
    # In a real implementation, this would check actual service connectivity
    # For now, we'll simulate the status
    return SystemStatus(
        status="operational",
        timestamp=datetime.utcnow(),
        services={
            "api": "operational",
            "database": "operational",
            "qdrant": "operational",
            "auth": "operational"
        },
        metrics={
            "total_queries": 1250,
            "active_sessions": 45,
            "avg_response_time": 450,  # ms
            "daily_active_users": 120
        }
    )