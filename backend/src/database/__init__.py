"""
Database initialization module.
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from .models import Base
import os
from dotenv import load_dotenv
import logging

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get database URL from environment variable
# Check for NEON_DATABASE_URL first (for Neon Postgres), then DATABASE_URL, then fallback to SQLite
DATABASE_URL = os.getenv("NEON_DATABASE_URL") or os.getenv("DATABASE_URL") or "sqlite:///./test.db"

# Create engine with proper configuration for Neon Postgres
if DATABASE_URL.startswith("postgresql"):
    # Configure for PostgreSQL/Neon with connection pooling settings
    engine = create_engine(
        DATABASE_URL,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,  # Validates connections before use
        pool_recycle=300,    # Recycle connections every 5 minutes
        echo=False  # Set to True for debugging SQL queries
    )
else:
    # Use default settings for SQLite
    engine = create_engine(DATABASE_URL)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    """
    Dependency function that provides database sessions.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def create_tables():
    """
    Create all database tables with error handling.
    """
    try:
        logger.info(f"Creating database tables using URL: {DATABASE_URL.replace('@', '***') if '@' in DATABASE_URL else DATABASE_URL}")
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")

        # Test the connection by trying to access the users table
        from sqlalchemy import text
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            logger.info("Database connection test successful")

    except Exception as e:
        logger.error(f"Error creating database tables: {str(e)}")
        raise