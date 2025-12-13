from sqlalchemy import Column, Integer, String, DateTime, Boolean, DECIMAL, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
import uuid
from db.database import Base


class ProgressRecord(Base):
    """
    ProgressRecord model for tracking user progress through course content.
    Records completion status, time spent, scores, and other progress metrics.
    """
    __tablename__ = "progress_records"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    content_id = Column(UUID(as_uuid=True), nullable=False)  # Can reference modules, lessons, or sections
    content_type = Column(String(50), nullable=False)  # 'module', 'lesson', 'section'
    status = Column(String(20), default='not_started', nullable=False)  # not_started, in_progress, completed, review_needed
    completion_percentage = Column(DECIMAL(5, 2), default=0.00)
    time_spent_minutes = Column(Integer, default=0)
    score = Column(DECIMAL(5, 2), nullable=True)  # 0.00 to 1.00, for assessments
    attempts_count = Column(Integer, default=0)
    last_accessed_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationship
    user = relationship("User", back_populates="progress_records")

    def __repr__(self):
        return f"<ProgressRecord(id={self.id}, user_id={self.user_id}, content_id={self.content_id}, status='{self.status}')>"


# Add relationship to User model
from .user import User  # Import here to avoid circular import
User.progress_records = relationship("ProgressRecord", order_by=ProgressRecord.created_at, back_populates="user")