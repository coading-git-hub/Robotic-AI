from sqlalchemy import Column, Integer, String, DateTime, Text, JSON, Boolean
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
import uuid
from db.database import Base


class ContentChunk(Base):
    """
    ContentChunk model for tracking book content that gets indexed in Qdrant.
    This model helps maintain synchronization between PostgreSQL and Qdrant.
    """
    __tablename__ = "content_chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    content_id = Column(UUID(as_uuid=True), nullable=False)  # References modules, lessons, or sections
    content_type = Column(String(50), nullable=False)  # 'module', 'lesson', 'section', 'exercise'
    text_content = Column(Text, nullable=False)
    source_module = Column(String(255), nullable=True)
    source_lesson = Column(String(255), nullable=True)
    token_count = Column(Integer, default=0)
    embedding_vector = Column(String(50), nullable=True)  # Store embedding ID reference
    is_indexed = Column(Boolean, default=False)  # Track if indexed in Qdrant
    indexed_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    def __repr__(self):
        return f"<ContentChunk(id={self.id}, content_type='{self.content_type}', content_id={self.content_id})>"