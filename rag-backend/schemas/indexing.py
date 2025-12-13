from pydantic import BaseModel
from uuid import UUID
from typing import Optional


class ContentIndexRequest(BaseModel):
    content_id: UUID
    content: str
    content_type: str  # 'module', 'lesson', 'section', 'exercise', etc.
    source_module: Optional[str] = None
    source_lesson: Optional[str] = None