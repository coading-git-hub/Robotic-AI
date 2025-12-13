from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.pool import AsyncAdaptedQueuePool
import os
from core.config import settings

import re

# Check if database URL is for SQLite
is_sqlite = settings.DATABASE_URL.startswith("sqlite")

# Create async engine with appropriate settings
connect_args = {}
if is_sqlite:
    # SQLite-specific settings
    connect_args = {
        "check_same_thread": False,  # Allow access from multiple threads
    }
else:
    # PostgreSQL-specific settings
    connect_args = {
        "server_settings": {
            "application_name": "physical-ai-rag",
            "tcp_keepalives_idle": "600",  # 10 minutes
            "tcp_keepalives_interval": "30",
            "tcp_keepalives_count": "5",
        }
    }

engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG,  # Set to True for SQL query logging in development
    poolclass=AsyncAdaptedQueuePool,
    pool_size=20 if not is_sqlite else 5,  # Smaller pool for SQLite
    max_overflow=30 if not is_sqlite else 10,
    pool_timeout=30,
    pool_recycle=3600 if not is_sqlite else 300,  # Shorter for SQLite
    pool_pre_ping=True,  # Verify connections before use
    connect_args=connect_args
)

# Create session maker for async sessions
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)

# Create base class for declarative models
Base = declarative_base()

async def get_db():
    """Dependency function to get database session"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()

async def init_db():
    """Initialize the database tables"""
    async with engine.begin() as conn:
        # Create all tables defined in models
        await conn.run_sync(Base.metadata.create_all)