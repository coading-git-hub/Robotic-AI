from sqlalchemy.ext.asyncio import AsyncSession
from typing import AsyncGenerator
from .database import AsyncSessionLocal

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency function that yields database sessions.
    This function is used as a FastAPI dependency to provide
    database sessions to route handlers.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()