from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import PointStruct
from typing import List, Optional, Dict, Any
from uuid import UUID
import logging
from core.config import settings

logger = logging.getLogger(__name__)


class QdrantService:
    def __init__(self):
        # Initialize Qdrant client
        if settings.QDRANT_URL and settings.QDRANT_API_KEY:
            self.client = QdrantClient(
                url=settings.QDRANT_URL,
                api_key=settings.QDRANT_API_KEY,
                timeout=10
            )
        else:
            # For local development
            self.client = QdrantClient(host="localhost", port=6333)

        self.collection_name = settings.QDRANT_COLLECTION_NAME
        self._ensure_collection_exists()

    def _ensure_collection_exists(self):
        """Ensure the collection exists with proper configuration."""
        try:
            # Check if collection exists
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]

            if self.collection_name not in collection_names:
                # Create collection with appropriate vector configuration
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=1536,  # Standard size for OpenAI embeddings
                        distance=models.Distance.COSINE
                    )
                )
                logger.info(f"Created Qdrant collection: {self.collection_name}")
            else:
                logger.info(f"Qdrant collection exists: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error ensuring collection exists: {e}")
            raise

    def search(self, query_text: str, limit: int = 5, filters: Optional[Dict[str, Any]] = None):
        """Perform semantic search in the Qdrant collection."""
        try:
            # In a real implementation, you would generate embeddings for the query text
            # For now, using a mock implementation
            search_filter = None
            if filters:
                # Convert filters to Qdrant filter format
                filter_conditions = []
                for key, value in filters.items():
                    filter_conditions.append(
                        models.FieldCondition(
                            key=key,
                            match=models.MatchValue(value=value)
                        )
                    )
                if filter_conditions:
                    search_filter = models.Filter(must=filter_conditions)

            # Perform search (in real implementation, this would use embeddings)
            results = self.client.search(
                collection_name=self.collection_name,
                query_text=query_text,  # This would be the embedding in real implementation
                limit=limit,
                query_filter=search_filter
            )

            return results
        except Exception as e:
            logger.error(f"Error performing search: {e}")
            raise

    def add_content(self, content_id: UUID, text: str, source_module: str = None, source_lesson: str = None, metadata: Dict[str, Any] = None):
        """Add content to the Qdrant collection."""
        try:
            # In a real implementation, you would generate embeddings for the text
            # For now, using a mock embedding
            import random
            embedding = [random.random() for _ in range(1536)]  # Mock embedding

            payload = {
                "text": text,
                "source_module": source_module or "",
                "source_lesson": source_lesson or "",
                "content_id": str(content_id),
                "metadata": metadata or {}
            }

            point = PointStruct(
                id=str(content_id),
                vector=embedding,
                payload=payload
            )

            self.client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )
        except Exception as e:
            logger.error(f"Error adding content to Qdrant: {e}")
            raise

    def delete_content(self, content_id: UUID):
        """Delete content from the Qdrant collection."""
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(
                    points=[str(content_id)]
                )
            )
        except Exception as e:
            logger.error(f"Error deleting content from Qdrant: {e}")
            raise

    def health(self):
        """Check health of the Qdrant service."""
        try:
            # Try to get collections as a basic health check
            self.client.get_collections()
            return {"status": "healthy", "service": "Qdrant"}
        except Exception as e:
            logger.error(f"Qdrant health check failed: {e}")
            return {"status": "unhealthy", "service": "Qdrant", "error": str(e)}


# Global instance
_qdrant_service = None


def get_qdrant_client():
    global _qdrant_service
    if _qdrant_service is None:
        _qdrant_service = QdrantService()
    return _qdrant_service