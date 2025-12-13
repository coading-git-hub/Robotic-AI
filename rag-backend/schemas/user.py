from pydantic import BaseModel, EmailStr
from typing import Optional
from datetime import datetime
from .auth import UserRole, ProgrammingLevel, RoboticsExperience


class User(BaseModel):
    """Schema for user model"""
    id: str
    username: str
    email: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    role: UserRole
    programming_level: Optional[ProgrammingLevel] = None
    robotics_experience: Optional[RoboticsExperience] = None
    created_at: datetime
    last_login_at: Optional[datetime] = None

    class Config:
        from_attributes = True