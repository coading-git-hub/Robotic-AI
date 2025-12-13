from pydantic import BaseModel, EmailStr, validator
from typing import Optional
from datetime import datetime
from enum import Enum


class UserRole(str, Enum):
    """User roles in the system"""
    STUDENT = "student"
    INSTRUCTOR = "instructor"
    ADMIN = "admin"


class ProgrammingLevel(str, Enum):
    """Programming experience levels"""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


class RoboticsExperience(str, Enum):
    """Robotics experience levels"""
    NONE = "none"
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


class UserCreate(BaseModel):
    """Schema for user creation"""
    username: str
    email: EmailStr
    password: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    role: Optional[UserRole] = UserRole.STUDENT
    programming_level: Optional[ProgrammingLevel] = None
    robotics_experience: Optional[RoboticsExperience] = None

    @validator('username')
    def validate_username(cls, v):
        if len(v) < 3 or len(v) > 50:
            raise ValueError('Username must be between 3 and 50 characters')
        if not v.replace('_', '').replace('.', '').isalnum():
            raise ValueError('Username can only contain alphanumeric characters, dots, and underscores')
        return v

    @validator('password')
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        # Check for at least one uppercase, lowercase, and digit
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not any(c.islower() for c in v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain at least one digit')
        return v


class Token(BaseModel):
    """Schema for authentication tokens"""
    access_token: str
    refresh_token: str
    token_type: str
    expires_in: int  # in seconds


class TokenData(BaseModel):
    """Schema for token data"""
    username: Optional[str] = None


class UserResponse(BaseModel):
    """Schema for user response"""
    id: str
    username: str
    email: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    role: UserRole
    programming_level: Optional[ProgrammingLevel] = None
    robotics_experience: Optional[RoboticsExperience] = None
    created_at: datetime

    class Config:
        from_attributes = True