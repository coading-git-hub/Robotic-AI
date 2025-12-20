"""
Personalization services for content adaptation based on user background.
"""
from sqlalchemy.orm import Session
from typing import Dict, Any, Optional
from ..database.models import User, PersonalizationSession
import uuid
from datetime import datetime

class PersonalizationService:
    """
    Service class for personalization operations.
    """

    @classmethod
    def personalize_content_for_user(
        cls,
        db: Session,
        user_id: str,
        chapter_id: str,
        user_background: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Apply personalization to content based on user background.
        """
        # Get user background if not provided
        if not user_background:
            user = db.query(User).filter(User.id == user_id).first()
            if user:
                user_background = {
                    "software_background": user.software_background,
                    "hardware_background": user.hardware_background
                }

        # Determine content adaptations based on user background
        content_variants = cls._determine_content_variants(user_background)

        # Create a personalization session record
        session = PersonalizationSession(
            user_id=user_id,
            chapter_id=chapter_id,
            background_applied=user_background or {},
            personalization_applied=True
        )

        db.add(session)
        db.commit()
        db.refresh(session)

        return {
            "session_id": session.id,
            "chapter_id": chapter_id,
            "personalization_applied": True,
            "content_variants": content_variants
        }

    @classmethod
    def _determine_content_variants(cls, user_background: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Determine content variations based on user background.
        """
        if not user_background:
            # Default content for users without background info
            return {
                "difficulty_level": "intermediate",
                "examples": ["general"],
                "hardware_specific_content": False,
                "content_adaptations": {
                    "explanation_depth": "moderate",
                    "example_complexity": "moderate",
                    "prerequisites_mentioned": True
                }
            }

        software_level = user_background.get("software_background", "").lower()
        hardware_level = user_background.get("hardware_background", "").lower()

        # Determine difficulty based on software background
        if "beginner" in software_level:
            difficulty_level = "beginner"
            explanation_depth = "detailed"
            example_complexity = "simple"
        elif any(level in software_level for level in ["frontend", "backend"]):
            difficulty_level = "intermediate"
            explanation_depth = "moderate"
            example_complexity = "moderate"
        elif "ai" in software_level:
            difficulty_level = "advanced"
            explanation_depth = "concise"
            example_complexity = "complex"
        else:
            difficulty_level = "intermediate"
            explanation_depth = "moderate"
            example_complexity = "moderate"

        # Determine hardware-specific content
        hardware_specific_content = any(hw in hardware_level for hw in ["gpu", "high-end"])

        # Determine appropriate examples
        if "beginner" in software_level:
            examples = ["simplified", "practical", "step-by-step"]
        elif "ai" in software_level:
            examples = ["advanced", "algorithm-focused", "performance-oriented"]
        else:
            examples = ["balanced", "practical"]

        return {
            "difficulty_level": difficulty_level,
            "examples": examples,
            "hardware_specific_content": hardware_specific_content,
            "content_adaptations": {
                "explanation_depth": explanation_depth,
                "example_complexity": example_complexity,
                "prerequisites_mentioned": "beginner" in software_level,
                "advanced_concepts_introduced": "ai" in software_level or "advanced" in hardware_level
            }
        }

    @classmethod
    def get_personalization_history(
        cls,
        db: Session,
        user_id: str
    ) -> list:
        """
        Get personalization history for a user.
        """
        sessions = db.query(PersonalizationSession).filter(
            PersonalizationSession.user_id == user_id
        ).order_by(PersonalizationSession.created_at.desc()).limit(20).all()

        return [
            {
                "session_id": session.id,
                "chapter_id": session.chapter_id,
                "background_applied": session.background_applied,
                "personalization_applied": session.personalization_applied,
                "created_at": session.created_at.isoformat()
            }
            for session in sessions
        ]