import openai
from typing import Dict, Any, List
import logging
from core.config import settings

logger = logging.getLogger(__name__)


class OpenAIService:
    def __init__(self):
        # Initialize OpenAI client with API key
        self.client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = settings.OPENAI_MODEL

    def generate_response(self, prompt: str, max_tokens: int = 500, temperature: float = 0.7):
        """Generate response using OpenAI API."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            raise

    def generate_embedding(self, text: str):
        """Generate embedding for the given text."""
        try:
            response = self.client.embeddings.create(
                input=text,
                model="text-embedding-ada-002"  # Standard embedding model
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise


# Global instance
_openai_service = None


def get_openai_client():
    global _openai_service
    if _openai_service is None:
        _openai_service = OpenAIService()
    return _openai_service