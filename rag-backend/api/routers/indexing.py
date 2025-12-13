from fastapi import APIRouter, Depends, HTTPException, status
from typing import Optional
from uuid import UUID
import logging

from models.user import User
from db.session import get_db
from utils.dependencies import get_current_user
from core.content_processor import get_content_processor
from schemas.indexing import ContentIndexRequest

router = APIRouter()

logger = logging.getLogger(__name__)


@router.post("/index")
async def index_content(
    index_request: ContentIndexRequest,
    current_user: User = Depends(get_current_user),  # Only authenticated users can index content
    db=Depends(get_db)
):
    """
    Index content for RAG search. This endpoint processes content and adds it to both
    PostgreSQL and Qdrant for semantic search capabilities.
    """
    try:
        # Check if user has admin/instructor privileges
        if current_user.role not in ["admin", "instructor"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only admin and instructor users can index content"
            )

        content_processor = get_content_processor()

        # Process and store content
        await content_processor.process_and_store_content(
            content=index_request.content,
            content_id=index_request.content_id,
            content_type=index_request.content_type,
            source_module=index_request.source_module,
            source_lesson=index_request.source_lesson
        )

        return {
            "message": "Content indexed successfully",
            "content_id": index_request.content_id,
            "chunks_processed": len(content_processor.chunk_content(
                index_request.content,
                index_request.content_id,
                index_request.source_module,
                index_request.source_lesson
            ))
        }

    except Exception as e:
        logger.error(f"Error indexing content: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to index content"
        )


@router.put("/index/{content_id}")
async def update_indexed_content(
    content_id: UUID,
    index_request: ContentIndexRequest,
    current_user: User = Depends(get_current_user),
    db=Depends(get_db)
):
    """
    Update previously indexed content.
    """
    try:
        # Check if user has admin/instructor privileges
        if current_user.role not in ["admin", "instructor"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only admin and instructor users can update indexed content"
            )

        content_processor = get_content_processor()

        # Update content
        await content_processor.update_content(
            content_id=content_id,
            new_content=index_request.content,
            content_type=index_request.content_type,
            source_module=index_request.source_module,
            source_lesson=index_request.source_lesson
        )

        return {
            "message": "Content updated and re-indexed successfully",
            "content_id": content_id
        }

    except Exception as e:
        logger.error(f"Error updating indexed content: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update indexed content"
        )


@router.delete("/index/{content_id}")
async def delete_indexed_content(
    content_id: UUID,
    current_user: User = Depends(get_current_user),
    db=Depends(get_db)
):
    """
    Delete indexed content from both PostgreSQL and Qdrant.
    """
    try:
        # Check if user has admin/instructor privileges
        if current_user.role not in ["admin", "instructor"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only admin and instructor users can delete indexed content"
            )

        content_processor = get_content_processor()

        # Delete content
        await content_processor.delete_content(content_id)

        return {
            "message": "Content deleted from index successfully",
            "content_id": content_id
        }

    except Exception as e:
        logger.error(f"Error deleting indexed content: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete indexed content"
        )


@router.post("/index/batch")
async def batch_index_content(
    content_list: list[ContentIndexRequest],
    current_user: User = Depends(get_current_user),
    db=Depends(get_db)
):
    """
    Batch index multiple content items at once.
    """
    try:
        # Check if user has admin/instructor privileges
        if current_user.role not in ["admin", "instructor"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only admin and instructor users can batch index content"
            )

        content_processor = get_content_processor()
        results = []

        for index_request in content_list:
            try:
                await content_processor.process_and_store_content(
                    content=index_request.content,
                    content_id=index_request.content_id,
                    content_type=index_request.content_type,
                    source_module=index_request.source_module,
                    source_lesson=index_request.source_lesson
                )
                results.append({
                    "content_id": index_request.content_id,
                    "status": "success"
                })
            except Exception as e:
                logger.error(f"Error indexing content {index_request.content_id}: {e}")
                results.append({
                    "content_id": index_request.content_id,
                    "status": "error",
                    "error": str(e)
                })

        return {
            "message": "Batch indexing completed",
            "results": results
        }

    except Exception as e:
        logger.error(f"Error in batch indexing: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to perform batch indexing"
        )