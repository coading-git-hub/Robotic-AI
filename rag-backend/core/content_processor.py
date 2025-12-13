from typing import List, Dict, Any
from uuid import UUID
import logging
from core.config import settings
from models.content_chunk import ContentChunk
from db.session import get_db
from core.qdrant_client import get_qdrant_client

logger = logging.getLogger(__name__)


class ContentProcessor:
    """
    Service class for processing book content into chunks suitable for RAG system.
    """

    def __init__(self):
        self.qdrant_client = get_qdrant_client()
        self.chunk_size = settings.CONTENT_CHUNK_SIZE  # Default to 1000 characters
        self.overlap_size = settings.CONTENT_CHUNK_OVERLAP  # Default to 100 characters

    def chunk_content(self, content: str, content_id: UUID, source_module: str = None, source_lesson: str = None) -> List[Dict[str, Any]]:
        """
        Split content into overlapping chunks for RAG indexing.
        """
        chunks = []
        start = 0

        while start < len(content):
            end = start + self.chunk_size

            # If we're near the end, make sure to include the rest
            if end > len(content):
                end = len(content)
            else:
                # Try to break at sentence boundary if possible
                while end < len(content) and content[end] not in '.!? \n' and end - start < self.chunk_size + 50:
                    end += 1
                if end == start + self.chunk_size + 50:  # If no sentence boundary found, just break at max
                    end = start + self.chunk_size

            chunk_text = content[start:end].strip()

            if len(chunk_text) > 0:  # Only add non-empty chunks
                chunk = {
                    'content_id': content_id,
                    'text_content': chunk_text,
                    'source_module': source_module,
                    'source_lesson': source_lesson,
                    'token_count': len(chunk_text.split())  # Rough token count
                }
                chunks.append(chunk)

            # Move start position, with overlap
            start = end - self.overlap_size
            if start >= end:  # Prevent infinite loop if overlap is too large
                start = end

        return chunks

    async def process_and_store_content(self, content: str, content_id: UUID, content_type: str,
                                      source_module: str = None, source_lesson: str = None):
        """
        Process content into chunks and store in both PostgreSQL and Qdrant.
        """
        try:
            # Chunk the content
            chunks = self.chunk_content(content, content_id, source_module, source_lesson)

            async with get_db() as db:
                for chunk_data in chunks:
                    # Create content chunk in PostgreSQL
                    content_chunk = ContentChunk(
                        content_id=content_id,
                        content_type=content_type,
                        text_content=chunk_data['text_content'],
                        source_module=source_module,
                        source_lesson=source_lesson,
                        token_count=chunk_data['token_count']
                    )

                    db.add(content_chunk)
                    await db.commit()
                    await db.refresh(content_chunk)

                    # Add to Qdrant for semantic search
                    try:
                        self.qdrant_client.add_content(
                            content_id=content_chunk.id,
                            text=content_chunk.text_content,
                            source_module=content_chunk.source_module,
                            source_lesson=content_chunk.source_lesson,
                            metadata={
                                "content_type": content_type,
                                "original_content_id": str(content_id)
                            }
                        )

                        # Mark as indexed in PostgreSQL
                        content_chunk.is_indexed = True
                        await db.commit()
                    except Exception as qe:
                        logger.error(f"Failed to index chunk {content_chunk.id} in Qdrant: {qe}")
                        # Continue with other chunks even if one fails to index

        except Exception as e:
            logger.error(f"Error processing content {content_id}: {e}")
            raise

    async def update_content(self, content_id: UUID, new_content: str, content_type: str,
                           source_module: str = None, source_lesson: str = None):
        """
        Update existing content by deleting old chunks and creating new ones.
        """
        try:
            # First, delete existing chunks from Qdrant
            # In a real implementation, we'd query for chunks with this content_id and delete them
            # For now, we'll just process the new content

            # Delete existing chunks in PostgreSQL for this content_id
            async with get_db() as db:
                # This would require a query to find and delete existing chunks
                # Then call process_and_store_content with the new content
                await self.process_and_store_content(
                    new_content, content_id, content_type, source_module, source_lesson
                )

        except Exception as e:
            logger.error(f"Error updating content {content_id}: {e}")
            raise

    async def delete_content(self, content_id: UUID):
        """
        Delete content chunks from both PostgreSQL and Qdrant.
        """
        try:
            # In a real implementation, we'd find all chunks for this content_id
            # and delete them from both PostgreSQL and Qdrant
            # For now, this is a placeholder
            async with get_db() as db:
                # Query and delete from PostgreSQL
                # Delete from Qdrant
                pass
        except Exception as e:
            logger.error(f"Error deleting content {content_id}: {e}")
            raise


# Global instance
_content_processor = None


def get_content_processor():
    global _content_processor
    if _content_processor is None:
        _content_processor = ContentProcessor()
    return _content_processor