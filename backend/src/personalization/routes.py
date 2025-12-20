"""
Personalization API routes for content adaptation based on user background.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from ..database import get_db
from ..database.models import User, PersonalizationSession
from ..auth.routes import get_current_user
from .services import PersonalizationService

router = APIRouter()

# Pydantic models for request/response
class PersonalizeRequest(BaseModel):
    chapter_id: str
    user_background: Optional[Dict[str, Any]] = None  # Will be extracted from user profile in real implementation

class PersonalizeResponse(BaseModel):
    session_id: str
    chapter_id: str
    personalization_applied: bool
    content_variants: Dict[str, Any]

class PersonalizationHistoryResponse(BaseModel):
    sessions: List[Dict[str, Any]]

@router.post("/personalize", response_model=PersonalizeResponse)
def personalize_content(
    request: PersonalizeRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Trigger content personalization for a chapter based on user background.
    """
    try:
        # Use the personalization service to handle the logic
        result = PersonalizationService.personalize_content_for_user(
            db=db,
            user_id=current_user.id,
            chapter_id=request.chapter_id,
            user_background=request.user_background
        )

        response = PersonalizeResponse(
            session_id=result["session_id"],
            chapter_id=result["chapter_id"],
            personalization_applied=result["personalization_applied"],
            content_variants=result["content_variants"]
        )

        return response
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred during personalization"
        )

@router.get("/history", response_model=PersonalizationHistoryResponse)
def get_personalization_history(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get user's personalization history.
    """
    try:
        # Get personalization history for the current user
        history = PersonalizationService.get_personalization_history(
            db=db,
            user_id=current_user.id
        )

        response = PersonalizationHistoryResponse(
            sessions=history
        )

        return response
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while fetching personalization history"
        )