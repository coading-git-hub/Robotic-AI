from fastapi import APIRouter, Depends, HTTPException, status
from typing import List, Optional
from uuid import UUID
from sqlalchemy import select
import logging

from models.user import User
from models.progress_record import ProgressRecord
from schemas.progress import ProgressCreate, ProgressUpdate, ProgressResponse
from db.session import get_db
from utils.dependencies import get_current_user

router = APIRouter()

logger = logging.getLogger(__name__)


@router.post("/", response_model=ProgressResponse)
async def create_progress_record(
    progress_data: ProgressCreate,
    current_user: User = Depends(get_current_user),
    db=Depends(get_db)
):
    """
    Create a new progress record for the authenticated user.
    """
    try:
        # Check if a record already exists for this content_id and user
        existing_record = await db.execute(
            select(ProgressRecord)
            .where(ProgressRecord.user_id == current_user.id)
            .where(ProgressRecord.content_id == progress_data.content_id)
        )
        existing = existing_record.scalar_one_or_none()

        if existing:
            # Update existing record
            for field, value in progress_data.dict(exclude_unset=True).items():
                setattr(existing, field, value)
            progress_record = existing
        else:
            # Create new record
            progress_record = ProgressRecord(
                user_id=current_user.id,
                content_id=progress_data.content_id,
                content_type=progress_data.content_type,
                module=progress_data.module,
                lesson=progress_data.lesson,
                completed=progress_data.completed,
                progress_percentage=progress_data.progress_percentage,
                time_spent=progress_data.time_spent
            )
            db.add(progress_record)

        await db.commit()
        await db.refresh(progress_record)

        return ProgressResponse.from_orm(progress_record)

    except Exception as e:
        await db.rollback()
        logger.error(f"Error creating progress record: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create progress record"
        )


@router.get("/{progress_id}", response_model=ProgressResponse)
async def get_progress_record(
    progress_id: UUID,
    current_user: User = Depends(get_current_user),
    db=Depends(get_db)
):
    """
    Get a specific progress record.
    Users can only access their own records.
    """
    progress_record = await db.get(ProgressRecord, progress_id)
    if not progress_record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Progress record not found"
        )

    # Check if the record belongs to the current user
    if progress_record.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this progress record"
        )

    return ProgressResponse.from_orm(progress_record)


@router.get("/", response_model=List[ProgressResponse])
async def get_user_progress(
    current_user: User = Depends(get_current_user),
    content_type: Optional[str] = None,
    module: Optional[str] = None,
    completed: Optional[bool] = None,
    db=Depends(get_db)
):
    """
    Get progress records for the authenticated user with optional filters.
    """
    try:
        query = select(ProgressRecord).where(ProgressRecord.user_id == current_user.id)

        if content_type:
            query = query.where(ProgressRecord.content_type == content_type)
        if module:
            query = query.where(ProgressRecord.module == module)
        if completed is not None:
            query = query.where(ProgressRecord.completed == completed)

        query = query.order_by(ProgressRecord.updated_at.desc())

        results = await db.execute(query)
        progress_records = results.scalars().all()

        return [ProgressResponse.from_orm(record) for record in progress_records]

    except Exception as e:
        logger.error(f"Error fetching progress records: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch progress records"
        )


@router.put("/{progress_id}", response_model=ProgressResponse)
async def update_progress_record(
    progress_id: UUID,
    progress_update: ProgressUpdate,
    current_user: User = Depends(get_current_user),
    db=Depends(get_db)
):
    """
    Update a progress record.
    Users can only update their own records.
    """
    try:
        progress_record = await db.get(ProgressRecord, progress_id)
        if not progress_record:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Progress record not found"
            )

        # Check if the record belongs to the current user
        if progress_record.user_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to update this progress record"
            )

        # Update fields
        for field, value in progress_update.dict(exclude_unset=True).items():
            setattr(progress_record, field, value)

        await db.commit()
        await db.refresh(progress_record)

        return ProgressResponse.from_orm(progress_record)

    except Exception as e:
        await db.rollback()
        logger.error(f"Error updating progress record: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update progress record"
        )


@router.delete("/{progress_id}")
async def delete_progress_record(
    progress_id: UUID,
    current_user: User = Depends(get_current_user),
    db=Depends(get_db)
):
    """
    Delete a progress record.
    Users can only delete their own records.
    """
    try:
        progress_record = await db.get(ProgressRecord, progress_id)
        if not progress_record:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Progress record not found"
            )

        # Check if the record belongs to the current user
        if progress_record.user_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to delete this progress record"
            )

        await db.delete(progress_record)
        await db.commit()

        return {"message": "Progress record deleted successfully"}

    except Exception as e:
        await db.rollback()
        logger.error(f"Error deleting progress record: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete progress record"
        )


@router.get("/summary/{user_id}", response_model=dict)
async def get_user_progress_summary(
    user_id: UUID,
    current_user: User = Depends(get_current_user),
    db=Depends(get_db)
):
    """
    Get a summary of user progress across all content.
    Users can only access their own summary, admins can access any user's summary.
    """
    # Check permissions
    if current_user.id != user_id and current_user.role not in ["admin", "instructor"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this progress summary"
        )

    try:
        # Get all progress records for the user
        all_records = await db.execute(
            select(ProgressRecord)
            .where(ProgressRecord.user_id == user_id)
        )
        records = all_records.scalars().all()

        if not records:
            return {
                "total_records": 0,
                "completed_records": 0,
                "completion_percentage": 0.0,
                "total_time_spent": 0
            }

        total_records = len(records)
        completed_records = len([r for r in records if r.completed])
        total_time_spent = sum(r.time_spent or 0 for r in records)
        completion_percentage = (completed_records / total_records) * 100 if total_records > 0 else 0.0

        return {
            "total_records": total_records,
            "completed_records": completed_records,
            "completion_percentage": round(completion_percentage, 2),
            "total_time_spent": total_time_spent
        }

    except Exception as e:
        logger.error(f"Error fetching progress summary: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch progress summary"
        )