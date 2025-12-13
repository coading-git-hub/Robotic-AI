from pydantic import BaseModel
from typing import Optional, List
from uuid import UUID
from datetime import datetime


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[UUID] = None
    selected_text: Optional[str] = None  # For selected-text mode functionality


class ChatResponse(BaseModel):
    response: str
    session_id: UUID
    sources: List[dict]  # List of source information


class ChatSessionCreate(BaseModel):
    session_title: Optional[str] = None


class ChatSessionUpdate(BaseModel):
    session_title: Optional[str] = None


class ChatSessionResponse(BaseModel):
    id: UUID
    user_id: Optional[UUID]
    session_title: Optional[str]
    active: bool
    created_at: datetime
    updated_at: datetime
    last_activity_at: datetime

    class Config:
        from_attributes = True