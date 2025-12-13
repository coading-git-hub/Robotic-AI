from fastapi import APIRouter, Depends, HTTPException, status
from typing import List, Optional
from uuid import UUID
from sqlalchemy import select
import logging

from models.user import User
from models.progress_record import ProgressRecord
from schemas.assessment import AssessmentRequest, AssessmentResponse, AssessmentResult, GradebookEntry
from db.session import get_db
from utils.dependencies import get_current_user

router = APIRouter()

logger = logging.getLogger(__name__)


@router.post("/submit", response_model=AssessmentResponse)
async def submit_assessment(
    assessment_request: AssessmentRequest,
    current_user: User = Depends(get_current_user),
    db=Depends(get_db)
):
    """
    Submit assessment answers and get grading results.
    """
    try:
        # Validate assessment data
        if not assessment_request.answers:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No answers provided for assessment"
            )

        # Calculate score
        correct_answers = 0
        total_questions = len(assessment_request.answers)

        for answer in assessment_request.answers:
            # In a real implementation, this would compare against correct answers
            # For now, we'll use a placeholder
            if answer.user_answer.lower() == answer.correct_answer.lower() if answer.correct_answer else True:
                correct_answers += 1

        score = (correct_answers / total_questions) * 100 if total_questions > 0 else 0

        # Create assessment result
        assessment_result = AssessmentResult(
            assessment_id=assessment_request.assessment_id,
            user_id=current_user.id,
            score=score,
            total_questions=total_questions,
            correct_answers=correct_answers,
            time_taken=assessment_request.time_taken,
            answers=assessment_request.answers
        )

        # Update progress record
        progress_record = ProgressRecord(
            user_id=current_user.id,
            content_id=assessment_request.assessment_id,
            content_type="assessment",
            module=assessment_request.module,
            lesson=assessment_request.lesson,
            completed=True,
            progress_percentage=score,
            time_spent=assessment_request.time_taken
        )

        db.add(progress_record)
        await db.commit()

        return AssessmentResponse(
            assessment_id=assessment_request.assessment_id,
            user_id=current_user.id,
            score=score,
            message=f"Assessment completed with score: {score:.2f}%"
        )

    except Exception as e:
        await db.rollback()
        logger.error(f"Error submitting assessment: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to submit assessment"
        )


@router.get("/results/{assessment_id}", response_model=AssessmentResult)
async def get_assessment_result(
    assessment_id: UUID,
    current_user: User = Depends(get_current_user),
    db=Depends(get_db)
):
    """
    Get results for a specific assessment.
    Users can only access their own assessment results.
    """
    try:
        # In a real implementation, this would fetch the assessment result from a dedicated table
        # For now, we'll return a placeholder
        return AssessmentResult(
            assessment_id=assessment_id,
            user_id=current_user.id,
            score=0.0,
            total_questions=0,
            correct_answers=0,
            time_taken=0,
            answers=[]
        )
    except Exception as e:
        logger.error(f"Error fetching assessment result: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch assessment result"
        )


@router.get("/gradebook", response_model=List[GradebookEntry])
async def get_gradebook(
    current_user: User = Depends(get_current_user),
    student_id: Optional[UUID] = None,
    db=Depends(get_db)
):
    """
    Get gradebook entries.
    Regular users can only see their own grades.
    Instructors and admins can see grades for their students or all users.
    """
    try:
        # Check permissions
        if current_user.role in ["student"]:
            # Students can only see their own grades
            if student_id and student_id != current_user.id:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Students can only view their own grades"
                )
            target_user_id = current_user.id
        elif current_user.role in ["instructor", "admin"]:
            # Instructors and admins can see grades for specific students or all
            target_user_id = student_id if student_id else current_user.id
        else:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions to view gradebook"
            )

        # In a real implementation, this would fetch gradebook entries from a dedicated table
        # For now, we'll return placeholder data
        gradebook_entries = [
            GradebookEntry(
                assessment_id=UUID("12345678-1234-5678-1234-567812345678"),
                assessment_name="Sample Quiz",
                module="Week 1",
                score=85.0,
                max_score=100.0,
                date_completed="2025-12-13T10:00:00Z"
            )
        ]

        return gradebook_entries

    except Exception as e:
        logger.error(f"Error fetching gradebook: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch gradebook"
        )


@router.get("/analytics", response_model=dict)
async def get_assessment_analytics(
    current_user: User = Depends(get_current_user),
    assessment_id: Optional[UUID] = None,
    module: Optional[str] = None,
    db=Depends(get_db)
):
    """
    Get analytics for assessments.
    Instructors and admins can access detailed analytics.
    """
    try:
        # Check permissions
        if current_user.role not in ["instructor", "admin"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only instructors and admins can access assessment analytics"
            )

        # In a real implementation, this would aggregate assessment data
        # For now, return placeholder analytics
        analytics = {
            "total_submissions": 0,
            "average_score": 0.0,
            "highest_score": 0.0,
            "lowest_score": 0.0,
            "completion_rate": 0.0,
            "analytics_generated_at": "2025-12-13T10:00:00Z"
        }

        return analytics

    except Exception as e:
        logger.error(f"Error fetching assessment analytics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch assessment analytics"
        )