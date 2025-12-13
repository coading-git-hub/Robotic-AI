from pydantic import BaseModel
from uuid import UUID
from datetime import datetime
from typing import Optional, List, Dict, Any


class ContentCreateRequest(BaseModel):
    content_id: UUID
    content: str
    content_type: str  # 'module', 'lesson', 'section', 'exercise', 'quiz', etc.
    source_module: Optional[str] = None
    source_lesson: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ContentUpdateRequest(BaseModel):
    content: str
    content_type: str
    source_module: Optional[str] = None
    source_lesson: Optional[str] = None


class ContentResponse(BaseModel):
    content_id: UUID
    status: str
    message: str
    indexed_at: str


class ContentValidationResponse(BaseModel):
    is_valid: bool
    issues: List[str]
    quality_score: float
    suggestions: List[str]


class ContentStatsResponse(BaseModel):
    total_chunks: int
    total_modules: int
    total_lessons: int
    indexed_at: str


class BatchContentItem(BaseModel):
    content_id: UUID
    content: str
    content_type: str
    source_module: Optional[str] = None
    source_lesson: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class BatchContentRequest(BaseModel):
    content_list: List[BatchContentItem]