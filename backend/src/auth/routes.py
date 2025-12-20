"""
Authentication API routes for user registration, login, and profile management.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from pydantic import BaseModel, EmailStr
from typing import Optional
from ..database import get_db
from ..database.models import User
from .services import AuthService
import jwt
from datetime import datetime, timedelta
import os
import uuid
from dotenv import load_dotenv

load_dotenv()

security = HTTPBearer()

# JWT configuration
SECRET_KEY = os.getenv("BETTER_AUTH_SECRET", "VBq3LTV3RMARjnHiqaht8zXB0hoSYk5Q")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """
    Create a JWT access token.
    """
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def verify_token(token: str):
    """
    Verify a JWT token and return the payload.
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security), db: Session = Depends(get_db)):
    """
    Get current user from JWT token.
    """
    token = credentials.credentials
    payload = verify_token(token)
    user_id: str = payload.get("sub")
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = AuthService.get_user_by_id(db, user_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user

router = APIRouter()

# Pydantic models for request/response
class UserCreateRequest(BaseModel):
    email: EmailStr
    password: str
    name: Optional[str] = None

class BetterAuthSignupRequest(BaseModel):
    email: EmailStr
    password: str
    name: Optional[str] = None

class UserUpdateRequest(BaseModel):
    name: Optional[str] = None
    software_background: Optional[str] = None
    hardware_background: Optional[str] = None

class UserResponse(BaseModel):
    id: str
    email: str
    name: Optional[str]
    software_background: Optional[str]
    hardware_background: Optional[str]
    created_at: str
    updated_at: str
    background_updated_at: Optional[str]

    class Config:
        from_attributes = True

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class LoginResponse(BaseModel):
    user: UserResponse
    token: str

class BetterAuthUserResponse(BaseModel):
    id: str
    email: str
    name: Optional[str]
    createdAt: str
    updatedAt: str

class BetterAuthSessionResponse(BaseModel):
    id: str
    userId: str
    token: str
    expiresAt: str

class BetterAuthSignupResponse(BaseModel):
    user: BetterAuthUserResponse
    session: BetterAuthSessionResponse

# Routes
@router.post("/signup", response_model=BetterAuthSignupResponse)
def signup(user_data: UserCreateRequest, db: Session = Depends(get_db)):
    """
    Register a new user with background information.
    """
    try:
        # Create user
        user = AuthService.create_user(
            db=db,
            email=user_data.email,
            password=user_data.password,
            name=user_data.name,
            software_background=None,  # Remove custom fields for better-auth compatibility
            hardware_background=None
        )

        # Create JWT token
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        token_data = {"sub": user.id, "email": user.email}
        token = create_access_token(data=token_data, expires_delta=access_token_expires)

        # Format response like better-auth
        user_response = BetterAuthUserResponse(
            id=user.id,
            email=user.email,
            name=user.name,
            createdAt=user.created_at.isoformat() if user.created_at else None,
            updatedAt=user.updated_at.isoformat() if user.updated_at else None
        )

        session_response = BetterAuthSessionResponse(
            id=str(uuid.uuid4()),  # Generate session id
            userId=user.id,
            token=token,
            expiresAt=(datetime.utcnow() + access_token_expires).isoformat()
        )

        return BetterAuthSignupResponse(user=user_response, session=session_response)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during user registration: {str(e)}"
        )

@router.post("/sign-up/email", response_model=BetterAuthSignupResponse)
def better_auth_signup(user_data: BetterAuthSignupRequest, db: Session = Depends(get_db)):
    """
    Better-auth compatible signup endpoint.
    """
    try:
        # Create user
        user = AuthService.create_user(
            db=db,
            email=user_data.email,
            password=user_data.password,
            name=user_data.name,
            software_background=None,
            hardware_background=None
        )

        # Create JWT token
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        token_data = {"sub": user.id, "email": user.email}
        token = create_access_token(data=token_data, expires_delta=access_token_expires)

        # Format response like better-auth
        user_response = BetterAuthUserResponse(
            id=user.id,
            email=user.email,
            name=user.name,
            createdAt=user.created_at.isoformat() if user.created_at else None,
            updatedAt=user.updated_at.isoformat() if user.updated_at else None
        )

        session_response = BetterAuthSessionResponse(
            id=str(uuid.uuid4()),  # Generate session id
            userId=user.id,
            token=token,
            expiresAt=(datetime.utcnow() + access_token_expires).isoformat()
        )

        return BetterAuthSignupResponse(user=user_response, session=session_response)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during user registration: {str(e)}"
        )

@router.post("/signin", response_model=LoginResponse)
def signin(login_data: LoginRequest, db: Session = Depends(get_db)):
    """
    Authenticate user and return user data with token.
    """
    user = AuthService.authenticate_user(
        db=db,
        email=login_data.email,
        password=login_data.password
    )

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password"
        )

    # Create a proper JWT token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    token_data = {"sub": user.id, "email": user.email}
    token = create_access_token(data=token_data, expires_delta=access_token_expires)

    user_response = UserResponse(
        id=user.id,
        email=user.email,
        name=user.name,
        software_background=user.software_background,
        hardware_background=user.hardware_background,
        created_at=user.created_at.isoformat() if user.created_at else None,
        updated_at=user.updated_at.isoformat() if user.updated_at else None,
        background_updated_at=user.background_updated_at.isoformat() if user.background_updated_at else None
    )

    return LoginResponse(user=user_response, token=token)

@router.get("/profile", response_model=UserResponse)
def get_profile(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get current user's profile.
    """
    try:
        user_response = UserResponse(
            id=current_user.id,
            email=current_user.email,
            name=current_user.name,
            software_background=current_user.software_background,
            hardware_background=current_user.hardware_background,
            created_at=current_user.created_at.isoformat() if current_user.created_at else None,
            updated_at=current_user.updated_at.isoformat() if current_user.updated_at else None,
            background_updated_at=current_user.background_updated_at.isoformat() if current_user.background_updated_at else None
        )

        return user_response
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while fetching profile"
        )


@router.put("/profile", response_model=UserResponse)
def update_profile(
    profile_data: UserUpdateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Update user's profile and background information.
    """
    try:
        # Update user profile information using the auth service
        updated_user = AuthService.update_user_profile(
            db=db,
            user_id=current_user.id,
            name=profile_data.name,
            software_background=profile_data.software_background,
            hardware_background=profile_data.hardware_background
        )

        if not updated_user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )

        # Return updated user data
        user_response = UserResponse(
            id=updated_user.id,
            email=updated_user.email,
            name=updated_user.name,
            software_background=updated_user.software_background,
            hardware_background=updated_user.hardware_background,
            created_at=updated_user.created_at.isoformat() if updated_user.created_at else None,
            updated_at=updated_user.updated_at.isoformat() if updated_user.updated_at else None,
            background_updated_at=updated_user.background_updated_at.isoformat() if updated_user.background_updated_at else None
        )

        return user_response
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while updating profile"
        )

