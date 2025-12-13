from fastapi import APIRouter, Depends, HTTPException, status
from typing import List, Optional
from uuid import UUID
import logging

from models.user import User
from db.session import get_db
from utils.dependencies import get_current_user
from core.content_manager import get_content_manager
from schemas.content_management import (
    ContentCreateRequest,
    ContentUpdateRequest,
    ContentResponse,
    ContentValidationResponse,
    ContentStatsResponse,
    BatchContentRequest
)

router = APIRouter()

logger = logging.getLogger(__name__)


@router.post("/", response_model=ContentResponse)
async def create_content(
    content_request: ContentCreateRequest,
    current_user: User = Depends(get_current_user),  # Only authenticated users can create content
    db=Depends(get_db)
):
    """
    Create new content and automatically index it for RAG search.
    """
    try:
        # Check if user has admin/instructor privileges
        if current_user.role not in ["admin", "instructor"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only admin and instructor users can create content"
            )

        content_manager = get_content_manager()

        # Validate content first
        validation = await content_manager.validate_content(content_request.content)
        if not validation["is_valid"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Content validation failed: {', '.join(validation['issues'])}"
            )

        # Create and index content
        result = await content_manager.create_content(
            content_id=content_request.content_id,
            content=content_request.content,
            content_type=content_request.content_type,
            source_module=content_request.source_module,
            source_lesson=content_request.source_lesson,
            metadata=content_request.metadata
        )

        return ContentResponse(
            content_id=content_request.content_id,
            status=result["status"],
            message=result["message"],
            indexed_at=result["indexed_at"]
        )

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error creating content: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create content"
        )


@router.put("/{content_id}", response_model=ContentResponse)
async def update_content(
    content_id: UUID,
    content_request: ContentUpdateRequest,
    current_user: User = Depends(get_current_user),
    db=Depends(get_db)
):
    """
    Update existing content and re-index it.
    """
    try:
        # Check if user has admin/instructor privileges
        if current_user.role not in ["admin", "instructor"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only admin and instructor users can update content"
            )

        content_manager = get_content_manager()

        # Validate content first
        validation = await content_manager.validate_content(content_request.content)
        if not validation["is_valid"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Content validation failed: {', '.join(validation['issues'])}"
            )

        # Update and re-index content
        result = await content_manager.update_content(
            content_id=content_id,
            new_content=content_request.content,
            content_type=content_request.content_type,
            source_module=content_request.source_module,
            source_lesson=content_request.source_lesson
        )

        return ContentResponse(
            content_id=content_id,
            status=result["status"],
            message=result["message"],
            indexed_at=result["indexed_at"]
        )

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error updating content: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update content"
        )


@router.delete("/{content_id}", response_model=ContentResponse)
async def delete_content(
    content_id: UUID,
    current_user: User = Depends(get_current_user),
    db=Depends(get_db)
):
    """
    Delete content from both PostgreSQL and Qdrant.
    """
    try:
        # Check if user has admin/instructor privileges
        if current_user.role not in ["admin", "instructor"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only admin and instructor users can delete content"
            )

        content_manager = get_content_manager()

        # Delete content
        result = await content_manager.delete_content(content_id)

        return ContentResponse(
            content_id=content_id,
            status=result["status"],
            message=result["message"],
            indexed_at=result["deleted_at"]
        )

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error deleting content: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete content"
        )


@router.post("/validate", response_model=ContentValidationResponse)
async def validate_content(
    content_request: ContentCreateRequest,
    current_user: User = Depends(get_current_user),
    db=Depends(get_db)
):
    """
    Validate content quality before indexing.
    """
    try:
        content_manager = get_content_manager()

        validation = await content_manager.validate_content(content_request.content)

        return ContentValidationResponse(**validation)

    except Exception as e:
        logger.error(f"Error validating content: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to validate content"
        )


@router.get("/stats", response_model=ContentStatsResponse)
async def get_content_stats(
    current_user: User = Depends(get_current_user),
    db=Depends(get_db)
):
    """
    Get statistics about indexed content.
    """
    try:
        # Only admin and instructor users can view content stats
        if current_user.role not in ["admin", "instructor"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only admin and instructor users can view content stats"
            )

        content_manager = get_content_manager()

        stats = await content_manager.get_content_stats()

        return ContentStatsResponse(
            total_chunks=stats["total_chunks"],
            total_modules=stats["total_modules"],
            total_lessons=stats["total_lessons"],
            indexed_at=stats["indexed_at"]
        )

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error getting content stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get content stats"
        )


@router.post("/sync", response_model=dict)
async def sync_content(
    current_user: User = Depends(get_current_user),
    db=Depends(get_db)
):
    """
    Synchronize content between PostgreSQL and Qdrant.
    """
    try:
        # Only admin users can perform content sync
        if current_user.role != "admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only admin users can perform content sync"
            )

        content_manager = get_content_manager()

        result = await content_manager.sync_content()

        return result

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error syncing content: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to sync content"
        )


@router.post("/batch", response_model=List[dict])
async def batch_process_content(
    batch_request: BatchContentRequest,
    current_user: User = Depends(get_current_user),
    db=Depends(get_db)
):
    """
    Process multiple content items in batch.
    """
    try:
        # Check if user has admin/instructor privileges
        if current_user.role not in ["admin", "instructor"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only admin and instructor users can perform batch content processing"
            )

        content_manager = get_content_manager()

        results = await content_manager.batch_process_content(batch_request.content_list)

        return results

    except Exception as e:
        logger.error(f"Error in batch content processing: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process batch content"
        )