from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional
import uuid

from db.session import get_db
from models.user import User
from utils.security import verify_token

security = HTTPBearer()

async def get_current_user(
    token: str = Depends(security),
    db: AsyncSession = Depends(get_db)
) -> User:
    """Dependency to get current authenticated user from token"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    token_data = verify_token(token.credentials)
    if token_data is None:
        raise credentials_exception

    user_id = token_data.get("sub")
    if user_id is None:
        raise credentials_exception

    try:
        user_id_uuid = uuid.UUID(user_id)
    except ValueError:
        raise credentials_exception

    # Query the user from the database
    from sqlalchemy import select
    query = select(User).where(User.id == user_id_uuid)
    result = await db.execute(query)
    user = result.scalars().first()

    if user is None:
        raise credentials_exception

    return user

async def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    """Dependency to get current active user (additional validation if needed)"""
    # Additional checks can be added here (e.g., account status, etc.)
    return current_user


async def get_current_user_optional(
    token: str = Depends(security),
    db: AsyncSession = Depends(get_db)
) -> Optional[User]:
    """Dependency to get current user if authenticated, otherwise return None"""
    try:
        token_data = verify_token(token.credentials)
        if token_data is None:
            return None

        user_id = token_data.get("sub")
        if user_id is None:
            return None

        try:
            user_id_uuid = uuid.UUID(user_id)
        except ValueError:
            return None

        # Query the user from the database
        from sqlalchemy import select
        query = select(User).where(User.id == user_id_uuid)
        result = await db.execute(query)
        user = result.scalars().first()

        return user
    except:
        # If any error occurs (e.g., no token provided), return None
        return None