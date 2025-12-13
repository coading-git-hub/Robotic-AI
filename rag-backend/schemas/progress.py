from pydantic import BaseModel
from uuid import UUID
from datetime import datetime
from typing import Optional


class ProgressBase(BaseModel):
    content_id: UUID
    content_type: str  # 'module', 'lesson', 'section', 'quiz', 'assignment', etc.
    module: Optional[str] = None
    lesson: Optional[str] = None
    completed: bool = False
    progress_percentage: float = 0.0
    time_spent: Optional[int] = None  # Time spent in seconds


class ProgressCreate(ProgressBase):
    pass


class ProgressUpdate(BaseModel):
    completed: Optional[bool] = None
    progress_percentage: Optional[float] = None
    time_spent: Optional[int] = None


class ProgressResponse(ProgressBase):
    id: UUID
    user_id: UUID
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True