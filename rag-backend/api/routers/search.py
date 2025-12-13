from fastapi import APIRouter, Depends, HTTPException, status
from typing import Optional, List
from uuid import UUID
import logging

from schemas.search import SearchRequest, SearchResponse
from models.user import User
from db.session import get_db
from utils.dependencies import get_current_user_optional
from core.qdrant_client import get_qdrant_client

router = APIRouter()

logger = logging.getLogger(__name__)


@router.post("/search", response_model=SearchResponse)
async def semantic_search(
    search_request: SearchRequest,
    current_user: Optional[User] = Depends(get_current_user_optional),
    db=Depends(get_db)
):
    """
    Perform semantic search across book content using Qdrant vector database.
    """
    try:
        qdrant_client = get_qdrant_client()

        # Perform semantic search in Qdrant
        search_results = qdrant_client.search(
            collection_name="book_content",  # Using default collection name
            query_text=search_request.query,
            limit=search_request.limit or 10,
            filters=search_request.filters  # Optional filters for specific modules/lessons
        )

        # Format results
        results = []
        for result in search_results:
            results.append({
                "id": result.id,
                "text": result.payload.get("text", ""),
                "source_module": result.payload.get("source_module", ""),
                "source_lesson": result.payload.get("source_lesson", ""),
                "score": result.score,
                "metadata": result.payload.get("metadata", {})
            })

        return SearchResponse(
            query=search_request.query,
            results=results,
            total=len(results)
        )

    except Exception as e:
        logger.error(f"Error performing semantic search: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to perform search"
        )


@router.get("/search/health")
async def search_health():
    """
    Health check for the search functionality.
    """
    try:
        qdrant_client = get_qdrant_client()
        # Perform a simple health check
        health = qdrant_client.health()
        return {"status": "healthy", "service": "Qdrant search"}
    except Exception as e:
        logger.error(f"Search health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Search service is not available"
        )