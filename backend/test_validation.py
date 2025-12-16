"""
Test suite for RAG retrieval validation pipeline.

This script contains unit and integration tests for the RAG retrieval validation pipeline.
"""
import unittest
import os
from unittest.mock import Mock, patch, MagicMock
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import the functions to be tested
try:
    from retrieval import (
        validate_query,
        sanitize_query,
        validate_query_embedding_compatibility,
        validate_metadata_completeness,
        validate_all_chunks_metadata,
        validate_content_relevance,
        validate_similarity_threshold,
        calculate_content_relevance_score,
        handle_empty_results,
        handle_low_confidence_results,
        calculate_retrieval_accuracy_metrics,
        validate_retrieval_accuracy_metrics,
        load_configuration,
        setup_logging,
        initialize_cohere_client,
        initialize_qdrant_client,
        embed_query,
        search_qdrant_vector_similarity,
        retrieve_chunks_with_metadata,
        run_end_to_end_pipeline_validation,
        test_system_stability_with_error_conditions
    )
    from qdrant_client.http import models
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you have the required dependencies installed and retrieval.py is in the same directory.")
    exit(1)


class TestQueryValidation(unittest.TestCase):
    """Test cases for query validation functions."""

    def test_validate_query_valid(self):
        """Test that valid queries pass validation."""
        valid_queries = [
            "What is AI?",
            "Explain machine learning",
            "How does neural network work"
        ]
        for query in valid_queries:
            with self.subTest(query=query):
                self.assertTrue(validate_query(query))

    def test_validate_query_invalid(self):
        """Test that invalid queries fail validation."""
        invalid_queries = [
            "",  # Empty
            "  ",  # Whitespace only
            "a",  # Too short
            "ab",  # Too short
            "x" * 1001,  # Too long
            "<script>alert('xss')</script>",  # Potential XSS
            "SELECT * FROM users",  # SQL injection
            "../etc/passwd",  # Path traversal
        ]
        for query in invalid_queries:
            with self.subTest(query=query[:50]):
                self.assertFalse(validate_query(query))

    def test_sanitize_query(self):
        """Test query sanitization."""
        test_cases = [
            ("  test query  ", "test query"),
            ("test    query", "test query"),
            ("test\t\nquery", "test query"),
            ("test query", "test query"),
        ]
        for input_query, expected in test_cases:
            with self.subTest(input_query=input_query):
                result = sanitize_query(input_query)
                self.assertEqual(result, expected)


class TestEmbeddingValidation(unittest.TestCase):
    """Test cases for embedding validation functions."""

    def test_validate_query_embedding_compatibility_valid(self):
        """Test that valid embeddings pass compatibility check."""
        # 768-dimensional vector (Cohere's multilingual model dimension)
        valid_embedding = [0.1] * 768
        self.assertTrue(validate_query_embedding_compatibility(valid_embedding))

    def test_validate_query_embedding_compatibility_invalid(self):
        """Test that invalid embeddings fail compatibility check."""
        invalid_embeddings = [
            [],  # Empty
            [0.1] * 100,  # Wrong dimension
            [0.1] * 1000,  # Wrong dimension
            None,  # None
            "not a list",  # Not a list
            [None] * 768,  # Contains non-numeric values
            ["string"] * 768,  # Contains non-numeric values
        ]
        for embedding in invalid_embeddings:
            with self.subTest(embedding=str(embedding)[:50]):
                self.assertFalse(validate_query_embedding_compatibility(embedding))


class TestMetadataValidation(unittest.TestCase):
    """Test cases for metadata validation functions."""

    def test_validate_metadata_completeness_valid(self):
        """Test that valid metadata passes completeness check."""
        valid_chunk = {
            'url': 'https://example.com',
            'title': 'Example Title',
            'chunk_index': 1,
            'content': 'Example content',
        }
        self.assertTrue(validate_metadata_completeness(valid_chunk))

    def test_validate_metadata_completeness_invalid(self):
        """Test that invalid metadata fails completeness check."""
        invalid_chunks = [
            {},  # Empty
            {'url': 'https://example.com', 'title': 'Example'},  # Missing chunk_index
            {'url': 'https://example.com', 'chunk_index': 1},  # Missing title
            {'title': 'Example', 'chunk_index': 1},  # Missing url
            {'url': '', 'title': 'Example', 'chunk_index': 1},  # Empty url
            {'url': 'https://example.com', 'title': '', 'chunk_index': 1},  # Empty title
            {'url': 'https://example.com', 'title': 'Example', 'chunk_index': -5},  # Invalid chunk_index
        ]
        for chunk in invalid_chunks:
            with self.subTest(chunk=chunk):
                self.assertFalse(validate_metadata_completeness(chunk))

    def test_validate_all_chunks_metadata(self):
        """Test validation of all chunks' metadata."""
        valid_chunks = [
            {'url': 'https://example1.com', 'title': 'Example 1', 'chunk_index': 1},
            {'url': 'https://example2.com', 'title': 'Example 2', 'chunk_index': 2},
        ]
        self.assertTrue(validate_all_chunks_metadata(valid_chunks))

        invalid_chunks = [
            {'url': 'https://example1.com', 'title': 'Example 1', 'chunk_index': 1},
            {'url': '', 'title': 'Example 2', 'chunk_index': 2},  # Invalid
        ]
        self.assertFalse(validate_all_chunks_metadata(invalid_chunks))


class TestRelevanceValidation(unittest.TestCase):
    """Test cases for relevance validation functions."""

    def test_validate_content_relevance(self):
        """Test content relevance validation using Qdrant similarity scores."""
        chunks = [
            {'content': 'test content 1', 'score': 0.7},  # Above threshold
            {'content': 'test content 2', 'score': 0.3},  # Below threshold
        ]

        # Mock Cohere client (not actually used in the new implementation)
        mock_cohere_client = Mock()

        result = validate_content_relevance(
            'test query',
            chunks,
            mock_cohere_client,
            relevance_threshold=0.5
        )

        # Should only return the relevant chunk based on Qdrant score
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['content'], 'test content 1')
        self.assertEqual(result[0]['relevance_score'], 0.7)  # Should use Qdrant score as relevance score


class TestConfiguration(unittest.TestCase):
    """Test cases for configuration loading."""

    def test_load_configuration_missing_vars(self):
        """Test configuration loading with missing environment variables."""
        # Temporarily unset required environment variables
        original_values = {
            'COHERE_API_KEY': os.environ.get('COHERE_API_KEY'),
            'QDRANT_URL': os.environ.get('QDRANT_URL'),
            'QDRANT_API_KEY': os.environ.get('QDRANT_API_KEY'),
        }

        # Unset the variables
        for key in original_values:
            if key in os.environ:
                del os.environ[key]

        try:
            with self.assertRaises(ValueError):
                load_configuration()
        finally:
            # Restore original values
            for key, value in original_values.items():
                if value is not None:
                    os.environ[key] = value


class TestHandleResults(unittest.TestCase):
    """Test cases for result handling functions."""

    def test_handle_empty_results(self):
        """Test handling of empty results."""
        config = {
            'similarity_threshold': 0.5,
            'top_k_results': 5
        }

        result = handle_empty_results("test query", config)

        self.assertEqual(result['result_status'], 'empty')
        self.assertEqual(result['query'], 'test query')
        self.assertEqual(result['similarity_threshold_used'], 0.5)
        self.assertEqual(result['top_k_requested'], 5)
        self.assertGreater(len(result['suggestions']), 0)

    def test_handle_low_confidence_results(self):
        """Test handling of low confidence results."""
        config = {
            'similarity_threshold': 0.8,
            'top_k_results': 5
        }

        chunks = [
            {'relevance_score': 0.3, 'content': 'test content'},
            {'relevance_score': 0.4, 'content': 'test content 2'}
        ]

        result = handle_low_confidence_results("test query", chunks, config)

        self.assertEqual(result['result_status'], 'low_confidence')
        self.assertEqual(result['query'], 'test query')
        self.assertEqual(result['highest_score'], 0.4)  # Max of the scores
        self.assertGreater(len(result['suggestions']), 0)


class TestIntegration(unittest.TestCase):
    """Integration tests for the pipeline."""

    @classmethod
    def setUpClass(cls):
        """Set up test configuration."""
        # Use environment variables or provide defaults for testing
        cls.config = {
            'top_k_results': int(os.getenv('TOP_K_RESULTS', 5)),
            'similarity_threshold': float(os.getenv('SIMILARITY_THRESHOLD', 0.3)),
            'retrieval_timeout': int(os.getenv('RETRIEVAL_TIMEOUT', 10)),
            'cohere_api_key': os.getenv('COHERE_API_KEY', 'test-key'),
            'qdrant_url': os.getenv('QDRANT_URL', 'http://localhost:6333'),
            'qdrant_api_key': os.getenv('QDRANT_API_KEY', 'test-key'),
        }

    @patch('retrieval.cohere.Client')
    @patch('retrieval.QdrantClient')
    def test_end_to_end_validation(self, mock_qdrant_client, mock_cohere_client):
        """Test end-to-end pipeline validation."""
        # Mock the clients
        mock_cohere_client_instance = Mock()
        mock_cohere_client.return_value = mock_cohere_client_instance

        mock_qdrant_client_instance = Mock()
        mock_qdrant_client.return_value = mock_qdrant_client_instance

        # Mock the embed method to return a test embedding
        mock_cohere_client_instance.embed.return_value = Mock()
        mock_cohere_client_instance.embed.return_value.embeddings = [[0.1] * 768]

        # Mock the search method to return test results
        mock_qdrant_client_instance.search.return_value = [
            models.ScoredPoint(
                id="test_id",
                version=1,
                score=0.8,
                payload={
                    'content': 'test content',
                    'url': 'https://example.com',
                    'title': 'Test Title',
                    'chunk_index': 1
                },
                vector=[0.1] * 768
            )
        ]

        sample_queries = ["What is AI?", "Explain machine learning"]

        results = run_end_to_end_pipeline_validation(
            mock_cohere_client_instance,
            mock_qdrant_client_instance,
            self.config,
            sample_queries
        )

        # Verify the results structure
        self.assertIn('total_queries', results)
        self.assertIn('successful_queries', results)
        self.assertIn('failed_queries', results)
        self.assertGreaterEqual(results['successful_queries'], 0)


if __name__ == '__main__':
    print("Running RAG Retrieval Validation Tests...")
    unittest.main(verbosity=2)