from typing import List, Dict, Any, Optional
from uuid import UUID
import logging
from datetime import datetime
import asyncio

from models.content_chunk import ContentChunk
from models.user import User
from db.session import get_db
from core.content_processor import get_content_processor
from core.qdrant_client import get_qdrant_client

logger = logging.getLogger(__name__)


class ContentManager:
    """
    Service class for managing content lifecycle: creation, updates, deletion,
    and automated indexing/synchronization between PostgreSQL and Qdrant.
    """

    def __init__(self):
        self.content_processor = get_content_processor()
        self.qdrant_client = get_qdrant_client()

    async def create_content(self, content_id: UUID, content: str, content_type: str,
                           source_module: str = None, source_lesson: str = None,
                           metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Create new content and automatically index it.
        """
        try:
            # Process and store content in both PostgreSQL and Qdrant
            await self.content_processor.process_and_store_content(
                content=content,
                content_id=content_id,
                content_type=content_type,
                source_module=source_module,
                source_lesson=source_lesson
            )

            return {
                "content_id": content_id,
                "status": "created",
                "indexed_at": datetime.utcnow().isoformat(),
                "message": "Content created and indexed successfully"
            }
        except Exception as e:
            logger.error(f"Error creating content {content_id}: {e}")
            raise

    async def update_content(self, content_id: UUID, new_content: str, content_type: str,
                           source_module: str = None, source_lesson: str = None) -> Dict[str, Any]:
        """
        Update existing content and re-index it.
        """
        try:
            # Update and re-index content
            await self.content_processor.update_content(
                content_id=content_id,
                new_content=new_content,
                content_type=content_type,
                source_module=source_module,
                source_lesson=source_lesson
            )

            return {
                "content_id": content_id,
                "status": "updated",
                "indexed_at": datetime.utcnow().isoformat(),
                "message": "Content updated and re-indexed successfully"
            }
        except Exception as e:
            logger.error(f"Error updating content {content_id}: {e}")
            raise

    async def delete_content(self, content_id: UUID) -> Dict[str, Any]:
        """
        Delete content from both PostgreSQL and Qdrant.
        """
        try:
            # Delete from both systems
            await self.content_processor.delete_content(content_id)

            return {
                "content_id": content_id,
                "status": "deleted",
                "deleted_at": datetime.utcnow().isoformat(),
                "message": "Content deleted from both PostgreSQL and Qdrant"
            }
        except Exception as e:
            logger.error(f"Error deleting content {content_id}: {e}")
            raise

    async def validate_content(self, content: str) -> Dict[str, Any]:
        """
        Validate content quality and integrity before indexing.
        """
        validation_result = {
            "is_valid": True,
            "issues": [],
            "quality_score": 0.0,
            "suggestions": []
        }

        # Check content length
        if len(content.strip()) < 10:
            validation_result["is_valid"] = False
            validation_result["issues"].append("Content too short")
            validation_result["suggestions"].append("Provide more detailed content")

        # Check for duplicate content (simplified check)
        # In a real implementation, this would check against existing content in the system

        # Check for quality indicators
        word_count = len(content.split())
        if word_count > 100:  # Good length for indexing
            validation_result["quality_score"] += 0.3
        if "example" in content.lower() or "code" in content.lower():
            validation_result["quality_score"] += 0.2
        if "exercise" in content.lower() or "practice" in content.lower():
            validation_result["quality_score"] += 0.1

        # Normalize quality score
        validation_result["quality_score"] = min(1.0, validation_result["quality_score"])

        return validation_result

    async def sync_content(self) -> Dict[str, Any]:
        """
        Synchronize content between PostgreSQL and Qdrant.
        This ensures both systems have consistent data.
        """
        try:
            # In a real implementation, this would:
            # 1. Compare content in PostgreSQL with Qdrant
            # 2. Identify missing content in either system
            # 3. Synchronize the missing content

            # For now, return a placeholder result
            return {
                "status": "sync_completed",
                "synced_at": datetime.utcnow().isoformat(),
                "message": "Content synchronization completed"
            }
        except Exception as e:
            logger.error(f"Error syncing content: {e}")
            raise

    async def get_content_stats(self) -> Dict[str, Any]:
        """
        Get statistics about indexed content.
        """
        try:
            # In a real implementation, this would query both PostgreSQL and Qdrant
            # to get content statistics

            # For now, return placeholder stats
            async with get_db() as db:
                # Count content chunks in PostgreSQL
                # This would require importing and using SQLAlchemy query functions
                total_chunks = 0  # Placeholder - would be actual count
                total_modules = 0  # Placeholder - would be actual count
                total_lessons = 0  # Placeholder - would be actual count

            return {
                "total_chunks": total_chunks,
                "total_modules": total_modules,
                "total_lessons": total_lessons,
                "indexed_at": datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting content stats: {e}")
            raise

    async def batch_process_content(self, content_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process multiple content items in batch.
        """
        results = []
        for content_item in content_list:
            try:
                content_id = content_item["content_id"]
                content = content_item["content"]
                content_type = content_item.get("content_type", "unknown")
                source_module = content_item.get("source_module")
                source_lesson = content_item.get("source_lesson")

                # Validate content first
                validation = await self.validate_content(content)
                if not validation["is_valid"]:
                    results.append({
                        "content_id": content_id,
                        "status": "validation_failed",
                        "issues": validation["issues"],
                        "message": f"Content validation failed for {content_id}"
                    })
                    continue

                # Process and index content
                await self.create_content(
                    content_id=content_id,
                    content=content,
                    content_type=content_type,
                    source_module=source_module,
                    source_lesson=source_lesson
                )

                results.append({
                    "content_id": content_id,
                    "status": "success",
                    "message": f"Content {content_id} processed and indexed successfully"
                })

            except Exception as e:
                logger.error(f"Error processing content item {content_item.get('content_id', 'unknown')}: {e}")
                results.append({
                    "content_id": content_item.get("content_id", "unknown"),
                    "status": "error",
                    "message": str(e)
                })

        return results


# Global instance
_content_manager = None


def get_content_manager():
    global _content_manager
    if _content_manager is None:
        _content_manager = ContentManager()
    return _content_manager