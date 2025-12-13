from sqlalchemy import Column, Integer, String, DateTime, Text, ForeignKey, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
import uuid
from db.database import Base


class Query(Base):
    """
    Query model for logging interactions with the RAG system.
    Records user queries, system responses, and interaction metadata.
    """
    __tablename__ = "queries"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=True)  # NULL for anonymous queries
    session_id = Column(UUID(as_uuid=True), ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False)
    input_text = Column(Text, nullable=False)
    selected_text = Column(Text, nullable=True)
    query_type = Column(String(50), default='general', nullable=False)  # general, selected_text, exercise_help, code_explanation
    response_text = Column(Text, nullable=True)
    retrieved_chunks = Column(JSON, nullable=True)  # Array of content chunk IDs
    confidence_score = Column(Integer, nullable=True)  # 0-100 scale
    response_time_ms = Column(Integer, nullable=True)
    feedback_rating = Column(Integer, nullable=True)  # 1-5 scale
    feedback_comment = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    user = relationship("User", back_populates="queries")
    session = relationship("ChatSession", back_populates="queries")

    def __repr__(self):
        return f"<Query(id={self.id}, user_id={self.user_id}, session_id={self.session_id}, query_type='{self.query_type}')>"


# Add relationship to User and ChatSession models
from .user import User  # Import here to avoid circular import
from .chat_session import ChatSession  # Import here to avoid circular import
User.queries = relationship("Query", order_by=Query.created_at, back_populates="user")
ChatSession.queries = relationship("Query", order_by=Query.created_at, back_populates="session")