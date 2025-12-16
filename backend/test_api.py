#!/usr/bin/env python3
"""
Test script to verify the RAG agent API implementation.
This script tests basic functionality without requiring external services.
"""

import sys
import os
import asyncio
from unittest.mock import Mock, patch, MagicMock

# Add the backend directory to the path so we can import rag_agent_api
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all necessary imports work without errors."""
    print("Testing imports...")
    try:
        import cohere
        import qdrant_client
        from fastapi import FastAPI
        from pydantic import BaseModel
        from dotenv import load_dotenv
        import uvicorn
        print("[PASS] All imports successful")
        return True
    except ImportError as e:
        print(f"[FAIL] Import error: {e}")
        return False

def test_rag_agent_api_load():
    """Test that rag_agent_api.py can be loaded without syntax errors."""
    print("Testing rag_agent_api.py load...")
    try:
        from rag_agent_api import (
            load_configuration,
            initialize_cohere_client,
            initialize_qdrant_client,
            validate_input,
            embed_query,
            get_context,
            prepare_agent_context,
            call_agent,
            validate_response,
            format_response,
            AgentQueryRequest,
            AgentQueryResponse,
            HealthCheckResponse,
            AgentValidateRequest,
            AgentValidateResponse,
            app
        )
        print("[PASS] rag_agent_api.py loaded successfully")
        return True
    except Exception as e:
        print(f"[FAIL] Error loading rag_agent_api.py: {e}")
        return False

def test_pydantic_models():
    """Test that Pydantic models are properly defined."""
    print("Testing Pydantic models...")
    try:
        from rag_agent_api import AgentQueryRequest, AgentQueryResponse, AgentValidateRequest

        # Test AgentQueryRequest model
        request = AgentQueryRequest(query="test query", selected_text="selected text")
        assert request.query == "test query"
        assert request.selected_text == "selected text"

        # Test with optional selected_text as None
        request2 = AgentQueryRequest(query="test query", selected_text=None)
        assert request2.query == "test query"
        assert request2.selected_text is None

        # Test AgentValidateRequest model
        validate_request = AgentValidateRequest(
            query="test query",
            context="test context",
            response="test response"
        )
        assert validate_request.query == "test query"
        assert validate_request.context == "test context"
        assert validate_request.response == "test response"

        print("[PASS] Pydantic models work correctly")
        return True
    except Exception as e:
        print(f"[FAIL] Error with Pydantic models: {e}")
        return False

def test_validation_functions():
    """Test validation functions."""
    print("Testing validation functions...")
    try:
        from rag_agent_api import validate_input

        # Test valid input
        result = validate_input("valid query", "selected text")
        assert result['is_valid'] == True
        assert result['cleaned_query'] == "valid query"
        assert result['cleaned_selected_text'] == "selected text"

        # Test query without selected text
        result2 = validate_input("valid query")
        assert result2['is_valid'] == True
        assert result2['cleaned_query'] == "valid query"
        assert result2['cleaned_selected_text'] is None

        # Test empty query
        result3 = validate_input("")
        assert result3['is_valid'] == False
        assert "cannot be empty" in result3['error_message']

        print("[PASS] Validation functions work correctly")
        return True
    except Exception as e:
        print(f"[FAIL] Error with validation functions: {e}")
        return False

def test_configuration_loading():
    """Test configuration loading with mocked environment variables."""
    print("Testing configuration loading...")
    try:
        from rag_agent_api import load_configuration
        import os

        # Temporarily set required environment variables
        original_values = {
            'COHERE_API_KEY': os.environ.get('COHERE_API_KEY'),
            'QDRANT_URL': os.environ.get('QDRANT_URL'),
            'QDRANT_API_KEY': os.environ.get('QDRANT_API_KEY'),
        }

        os.environ['COHERE_API_KEY'] = 'test-key'
        os.environ['QDRANT_URL'] = 'https://test.qdrant.com'
        os.environ['QDRANT_API_KEY'] = 'test-key'

        try:
            config = load_configuration()
            assert 'cohere_api_key' in config
            assert 'qdrant_url' in config
            assert 'qdrant_api_key' in config
            assert config['cohere_api_key'] == 'test-key'
            assert config['qdrant_url'] == 'https://test.qdrant.com'
            assert config['qdrant_api_key'] == 'test-key'

            print("[PASS] Configuration loading works correctly")
            return True
        finally:
            # Restore original values
            for key, value in original_values.items():
                if value is not None:
                    os.environ[key] = value
                elif key in os.environ:
                    del os.environ[key]
    except Exception as e:
        print(f"[FAIL] Error with configuration loading: {e}")
        return False

def run_all_tests():
    """Run all tests."""
    print("Starting RAG Agent API tests...\n")

    tests = [
        test_imports,
        test_rag_agent_api_load,
        test_pydantic_models,
        test_validation_functions,
        test_configuration_loading
    ]

    passed = 0
    total = len(tests)

    for test_func in tests:
        try:
            if test_func():
                passed += 1
            print()  # Add spacing
        except Exception as e:
            print(f"[FAIL] Test {test_func.__name__} failed with exception: {e}")
            print()

    print(f"Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("[SUCCESS] All tests passed! The RAG Agent API implementation is working correctly.")
        return True
    else:
        print("[WARNING] Some tests failed. Please review the implementation.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)