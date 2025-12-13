from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from uuid import UUID


class SearchRequest(BaseModel):
    query: str
    limit: Optional[int] = 10
    filters: Optional[Dict[str, Any]] = None  # For filtering by module, lesson, etc.


class SearchResult(BaseModel):
    id: UUID
    text: str
    source_module: str
    source_lesson: str
    score: float
    metadata: Dict[str, Any]


class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    total: int