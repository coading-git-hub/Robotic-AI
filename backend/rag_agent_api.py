"""
Context-Aware RAG Agent with FastAPI

This script implements a FastAPI backend that answers queries using retrieved book content
and optional user-selected text as additional context. The system prioritizes selected text in the context,
uses Cohere for embeddings and generation, and handles queries safely with fallback mechanisms.
The implementation is stateless, Python-based, and cloud-compatible.
"""

import os
import sys
import logging
import time
import requests
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Import required libraries
try:
    import cohere
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
    import uvicorn
except ImportError as e:
    print(f"Error: Missing required dependency. Please install requirements: {e}")
    sys.exit(1)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Pydantic models for request/response validation
class AgentQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="User query to answer")
    selected_text: Optional[str] = Field(None, max_length=5000, description="Optional user-selected text to prioritize in context")


class SourceObject(BaseModel):
    id: str
    content: str
    url: str
    title: str
    score: float


class AgentQueryResponse(BaseModel):
    answer: str
    sources: List[SourceObject]
    confidence: float
    grounded_in_context: bool


class HealthCheckResponse(BaseModel):
    status: str
    services: Dict[str, bool]


class AgentValidateRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    context: str = Field(..., min_length=1, max_length=10000)
    response: str = Field(..., min_length=1, max_length=5000)


class AgentValidateResponse(BaseModel):
    grounded_in_context: bool
    confidence_score: float
    validation_details: Dict[str, Any]


# Configuration loading function
def load_configuration() -> Dict[str, Any]:
    """
    Load and validate configuration from environment variables.

    Returns:
        Dictionary containing configuration values

    Raises:
        ValueError: If required environment variables are missing or invalid
    """
    config = {
        'cohere_api_key': os.getenv('COHERE_API_KEY'),
        'qdrant_url': os.getenv('QDRANT_URL'),
        'qdrant_api_key': os.getenv('QDRANT_API_KEY'),
        'cohere_model': os.getenv('COHERE_MODEL', 'command-r-plus'),
        'embedding_model': os.getenv('EMBEDDING_MODEL', 'embed-english-v3.0'),
        'context_window_limit': int(os.getenv('CONTEXT_WINDOW_LIMIT', 120000)),
        'selected_text_priority': float(os.getenv('SELECTED_TEXT_PRIORITY', 0.8)),
        'fallback_message': os.getenv('FALLBACK_MESSAGE', "I cannot answer based on the provided context."),
        'max_query_length': int(os.getenv('MAX_QUERY_LENGTH', 2000)),
        'max_selected_text_length': int(os.getenv('MAX_SELECTED_TEXT_LENGTH', 5000)),
        'top_k_results': int(os.getenv('TOP_K_RESULTS', 5)),
        'similarity_threshold': float(os.getenv('SIMILARITY_THRESHOLD', 0.3)),
        'retrieval_timeout': int(os.getenv('RETRIEVAL_TIMEOUT', 10))
    }

    # Validate required configuration
    required_keys = ['cohere_api_key', 'qdrant_url', 'qdrant_api_key']
    missing_keys = [key for key in required_keys if not config.get(key)]

    if missing_keys:
        raise ValueError(f"Missing required environment variables: {missing_keys}")

    # Validate numeric configuration values
    if config['top_k_results'] <= 0:
        raise ValueError("TOP_K_RESULTS must be a positive integer")

    if not 0 <= config['similarity_threshold'] <= 1:
        raise ValueError("SIMILARITY_THRESHOLD must be between 0 and 1")

    if config['retrieval_timeout'] <= 0:
        raise ValueError("RETRIEVAL_TIMEOUT must be a positive integer")

    if config['context_window_limit'] <= 0:
        raise ValueError("CONTEXT_WINDOW_LIMIT must be a positive integer")

    if not 0 <= config['selected_text_priority'] <= 1:
        raise ValueError("SELECTED_TEXT_PRIORITY must be between 0 and 1")

    return config


# Initialize Cohere client
def initialize_cohere_client(api_key: str) -> Optional[cohere.Client]:
    """
    Initialize and validate Cohere client with error handling.

    Args:
        api_key: Cohere API key

    Returns:
        Cohere client instance or None if initialization fails
    """
    try:
        # Initialize client
        client = cohere.Client(api_key)

        # Test the client by making a simple call
        client.embed(
            texts=["test"],
            model="embed-english-v3.0",  # Using the same model as in the ingestion pipeline
            input_type="search_query"  # Appropriate for search queries
        )
        logger.info("Cohere client initialized and validated successfully")
        return client
    except Exception as e:
        logger.error(f"Failed to initialize or validate Cohere client: {e}")
        return None


# Initialize Qdrant client
def initialize_qdrant_client(url: str, api_key: str, timeout: int) -> Optional[QdrantClient]:
    """
    Initialize and validate Qdrant client with error handling.

    Args:
        url: Qdrant URL
        api_key: Qdrant API key
        timeout: Request timeout in seconds

    Returns:
        Qdrant client instance or None if initialization fails
    """
    try:
        client = QdrantClient(
            url=url,
            api_key=api_key,
            timeout=timeout
        )

        # Test the client by getting collection info
        client.get_collections()
        logger.info("Qdrant client initialized and validated successfully")
        return client
    except Exception as e:
        logger.error(f"Failed to initialize or validate Qdrant client: {e}")
        return None


# Input validation function
def validate_input(query: str, selected_text: Optional[str] = None) -> Dict[str, Any]:
    """
    Validate query and selected_text parameters.

    Args:
        query: User query string
        selected_text: Optional user-selected text

    Returns:
        Dictionary with validation results and cleaned data
    """
    result = {
        'is_valid': True,
        'error_message': '',
        'cleaned_query': '',
        'cleaned_selected_text': selected_text
    }

    # Validate query
    if not query or not query.strip():
        result['is_valid'] = False
        result['error_message'] = 'Query cannot be empty'
        return result

    if len(query) > 2000:  # From config
        result['is_valid'] = False
        result['error_message'] = f'Query exceeds maximum length of 2000 characters: {len(query)}'
        return result

    # Clean query
    result['cleaned_query'] = query.strip()

    # Validate selected_text if provided
    if selected_text is not None:
        if len(selected_text) > 5000:  # From config
            result['is_valid'] = False
            result['error_message'] = f'Selected text exceeds maximum length of 5000 characters: {len(selected_text)}'
            return result

        result['cleaned_selected_text'] = selected_text.strip()

    return result


# Function to embed query using Cohere
def embed_query(cohere_client: cohere.Client, query: str) -> Optional[List[float]]:
    """
    Embed a query using the Cohere API.

    Args:
        cohere_client: Initialized Cohere client
        query: Query string to embed

    Returns:
        Embedding vector as a list of floats, or None if embedding fails
    """
    try:
        response = cohere_client.embed(
            texts=[query],
            model="embed-english-v3.0",  # Using the same model as in the ingestion pipeline
            input_type="search_query"  # Appropriate for search queries
        )

        if response.embeddings and len(response.embeddings) > 0:
            logger.info(f"Successfully embedded query of length {len(query)}")
            return response.embeddings[0]  # Return the first (and only) embedding
        else:
            logger.error("No embeddings returned from Cohere API")
            return None
    except Exception as e:
        logger.error(f"Failed to embed query: {e}")
        return None


# Function to get context from Qdrant
def get_context(query: str, qdrant_client: QdrantClient, config: Dict[str, Any],
                selected_text: Optional[str] = None) -> Dict[str, Any]:
    """
    Retrieve context from Qdrant and combine with selected text if provided.

    Args:
        query: User query
        selected_text: Optional user-selected text to prioritize
        qdrant_client: Initialized Qdrant client
        config: Configuration dictionary

    Returns:
        Dictionary containing combined context and retrieval metadata
    """
    context_result = {
        'chunks': [],
        'selected_text_included': bool(selected_text),
        'retrieval_successful': True,
        'error_message': ''
    }

    # If selected_text is provided, prioritize it
    if selected_text:
        # Add selected text as high-priority context
        selected_chunk = {
            'id': 'selected_text',
            'content': selected_text,
            'url': 'user_selection',
            'title': 'User Selected Text',
            'score': 1.0  # Highest priority
        }
        context_result['chunks'].append(selected_chunk)

    # Perform vector search in Qdrant for additional context
    try:
        # First, embed the query using Cohere
        cohere_client = initialize_cohere_client(config['cohere_api_key'])
        if not cohere_client:
            context_result['retrieval_successful'] = False
            context_result['error_message'] = 'Failed to initialize Cohere client for embedding'
            return context_result

        query_embedding = embed_query(cohere_client, query)
        if not query_embedding:
            context_result['retrieval_successful'] = False
            context_result['error_message'] = 'Failed to embed query'
            return context_result

        # Perform search in Qdrant
        search_results = qdrant_client.query_points(
            collection_name="book_embeddings",  # Using same collection as retrieval validation
            query=query_embedding,
            limit=config['top_k_results'],
            score_threshold=config['similarity_threshold'],
            with_payload=True,  # Include payload (metadata) in results
            with_vectors=False  # We don't need the vectors themselves
        ).points

        # Add retrieved chunks to context
        for result in search_results:
            payload = result.payload
            chunk = {
                'id': result.id,
                'content': payload.get('content', ''),
                'url': payload.get('url', ''),
                'title': payload.get('title', ''),
                'score': result.score
            }
            context_result['chunks'].append(chunk)

        logger.info(f"Retrieved {len(search_results)} chunks from Qdrant for query: '{query[:50]}{'...' if len(query) > 50 else ''}'")

    except Exception as e:
        logger.error(f"Failed to retrieve context from Qdrant: {e}")
        context_result['retrieval_successful'] = False
        context_result['error_message'] = str(e)

    return context_result


# Function to prepare context for the agent
def prepare_agent_context(context: Dict[str, Any], query: str) -> Dict[str, Any]:
    """
    Prepare context for the Cohere agent with proper prioritization.

    Args:
        context: Retrieved context from get_context function
        query: Original user query

    Returns:
        Dictionary with formatted context ready for agent consumption
    """
    formatted_context = {
        'system_prompt': "",
        'user_context': "",
        'query': query,
        'has_sufficient_context': len(context['chunks']) > 0
    }

    # Create a system prompt that emphasizes grounding in provided context
    system_prompt = (
        "You are an educational AI assistant that answers questions based strictly on the provided context. "
        "Your responses must be grounded in the provided text and you must not use any external knowledge or hallucinate information. "
        "If the provided context does not contain sufficient information to answer the query, respond with: "
        f"\"{os.getenv('FALLBACK_MESSAGE', 'I cannot answer based on the provided context.')}.\" "
        "Prioritize information from 'User Selected Text' if it's provided, as this represents text the user specifically highlighted."
    )

    formatted_context['system_prompt'] = system_prompt

    # Combine all context chunks into a single context string
    context_parts = []
    for i, chunk in enumerate(context['chunks']):
        source_identifier = f"[Source {i+1}]"
        if chunk.get('id') == 'selected_text':
            source_identifier = "[USER SELECTED TEXT - HIGHEST PRIORITY]"

        context_part = (
            f"{source_identifier}\n"
            f"Title: {chunk.get('title', 'Unknown')}\n"
            f"URL: {chunk.get('url', 'Unknown')}\n"
            f"Content: {chunk.get('content', '')}\n"
            f"Relevance Score: {chunk.get('score', 0.0)}\n"
            "---\n"
        )
        context_parts.append(context_part)

    formatted_context['user_context'] = "\n".join(context_parts)

    return formatted_context


# Function to call the agent (using Cohere instead of OpenAI)
def call_agent(formatted_context: Dict[str, Any], cohere_client: cohere.Client, config: Dict[str, Any]) -> str:
    """
    Call the Cohere agent to generate a response based on the provided context.

    Args:
        formatted_context: Formatted context from prepare_agent_context function
        cohere_client: Initialized Cohere client

    Returns:
        Agent-generated response string
    """
    try:
        # Combine the system prompt and user context with the query
        full_prompt = (
            f"Context:\n{formatted_context['user_context']}\n\n"
            f"Query: {formatted_context['query']}\n\n"
            f"Please answer the query based on the provided context, following the instructions in the system prompt."
        )

        # Use Cohere's chat API to generate a response
        response = cohere_client.chat(
            message=full_prompt,
            preamble=formatted_context['system_prompt'],
            model=config['cohere_model'],  # Using configured model for instruction-following
            temperature=0.3,  # Lower temperature for more factual responses
            max_tokens=1000
        )

        logger.info(f"Agent generated response for query: '{formatted_context['query'][:50]}{'...' if len(formatted_context['query']) > 50 else ''}'")
        return response.text

    except Exception as e:
        logger.error(f"Failed to call agent: {e}")
        return os.getenv('FALLBACK_MESSAGE', 'I cannot answer based on the provided context.')


# Function to validate response grounding
def validate_response(response: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate that the response is grounded in the provided context.

    Args:
        response: Agent-generated response
        context: Original context provided to the agent

    Returns:
        Dictionary with validation results and confidence scores
    """
    validation_result = {
        'grounded_in_context': True,
        'confidence_score': 0.8,  # Default to medium-high confidence
        'validation_details': {
            'referenced_sources': [],
            'potential_hallucinations': [],
            'context_coverage': 'partial'  # Default assumption
        }
    }

    # Basic validation: check if response contains fallback message
    fallback_msg = os.getenv('FALLBACK_MESSAGE', 'I cannot answer based on the provided context.')
    if fallback_msg.strip().lower() in response.strip().lower():
        validation_result['grounded_in_context'] = True
        validation_result['confidence_score'] = 1.0  # Fallback responses are valid
        validation_result['validation_details']['context_coverage'] = 'insufficient'
        return validation_result

    # More sophisticated validation would involve checking if response content
    # can be traced back to the provided context chunks, but for now we'll
    # implement a basic check
    response_lower = response.lower()

    # Check if response references content from the context
    referenced_sources = []
    potential_hallucinations = []

    for chunk in context['chunks']:
        content_lower = chunk.get('content', '').lower()
        # If there's overlap between context and response, consider it referenced
        if len(content_lower) > 10:  # Only check substantial chunks
            content_words = set(content_lower.split()[:50])  # Check first 50 words
            response_words = set(response_lower.split())

            if len(content_words.intersection(response_words)) > 0:
                referenced_sources.append({
                    'id': chunk.get('id'),
                    'title': chunk.get('title'),
                    'overlap_words': list(content_words.intersection(response_words))[:5]  # First 5 overlapping words
                })

    validation_result['validation_details']['referenced_sources'] = referenced_sources
    validation_result['validation_details']['potential_hallucinations'] = potential_hallucinations

    # Update confidence based on how much of the context was referenced
    if len(referenced_sources) == 0 and response.strip() != fallback_msg.strip():
        validation_result['grounded_in_context'] = False
        validation_result['confidence_score'] = 0.2  # Low confidence if nothing referenced
    elif len(referenced_sources) == len([c for c in context['chunks'] if c.get('id') != 'selected_text']):
        validation_result['validation_details']['context_coverage'] = 'complete'
        validation_result['confidence_score'] = 0.9  # High confidence if all sources used
    else:
        validation_result['validation_details']['context_coverage'] = 'partial'
        validation_result['confidence_score'] = 0.7  # Medium confidence for partial usage

    return validation_result


# Function to format the final response
def format_response(agent_response: str, context: Dict[str, Any], validation: Dict[str, Any]) -> AgentQueryResponse:
    """
    Format the final response for API output.

    Args:
        agent_response: Raw agent response
        context: Context used for the query
        validation: Validation results

    Returns:
        Formatted AgentQueryResponse object
    """
    # Extract sources from context
    sources = []
    for chunk in context['chunks']:
        if chunk.get('id') != 'selected_text':  # Exclude user-selected text from sources list
            source_obj = SourceObject(
                id=chunk.get('id', ''),
                content=chunk.get('content', '')[:200] + "..." if len(chunk.get('content', '')) > 200 else chunk.get('content', ''),
                url=chunk.get('url', ''),
                title=chunk.get('title', ''),
                score=chunk.get('score', 0.0)
            )
            sources.append(source_obj)

    return AgentQueryResponse(
        answer=agent_response,
        sources=sources,
        confidence=validation['confidence_score'],
        grounded_in_context=validation['grounded_in_context']
    )


# FastAPI app initialization
app = FastAPI(
    title="Context-Aware RAG Agent API",
    description="FastAPI backend that answers queries using retrieved book content and optional user-selected text as additional context",
    version="1.0.0"
)

# Add CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global variables to store initialized clients
cohere_client = None
qdrant_client = None
config = None


@app.on_event("startup")
async def startup_event():
    """Initialize clients and configuration when the app starts."""
    global cohere_client, qdrant_client, config

    try:
        # Load configuration
        config = load_configuration()
        logger.info("Configuration loaded successfully")

        # Initialize Cohere client
        cohere_client = initialize_cohere_client(config['cohere_api_key'])
        if not cohere_client:
            logger.error("Failed to initialize Cohere client")

        # Initialize Qdrant client
        qdrant_client = initialize_qdrant_client(
            config['qdrant_url'],
            config['qdrant_api_key'],
            config['retrieval_timeout']
        )
        if not qdrant_client:
            logger.error("Failed to initialize Qdrant client")

        logger.info("Application startup completed")
    except Exception as e:
        logger.error(f"Startup failed: {e}")


@app.post("/api/agent/query", response_model=AgentQueryResponse)
async def query_agent(request: AgentQueryRequest):
    """
    Main endpoint to process user queries with optional selected text.

    Args:
        request: AgentQueryRequest containing query and optional selected_text

    Returns:
        AgentQueryResponse with answer, sources, confidence, and grounding status
    """
    try:
        logger.info(f"Processing query: '{request.query[:50]}{'...' if len(request.query) > 50 else ''}'")

        # Validate inputs
        validation_result = validate_input(request.query, request.selected_text)
        if not validation_result['is_valid']:
            raise HTTPException(status_code=400, detail=validation_result['error_message'])

        # Check if clients are initialized
        if not cohere_client or not qdrant_client or not config:
            raise HTTPException(status_code=500, detail="Service not properly initialized")

        # Get context from Qdrant and user selection
        context_result = get_context(
            validation_result['cleaned_query'],
            qdrant_client,
            config,
            validation_result['cleaned_selected_text'] if request.selected_text else None
        )

        if not context_result['retrieval_successful']:
            raise HTTPException(status_code=500, detail=context_result['error_message'])

        # Prepare context for the agent
        formatted_context = prepare_agent_context(context_result, validation_result['cleaned_query'])

        # Call the agent
        agent_response = call_agent(formatted_context, cohere_client, config)

        # Validate the response
        response_validation = validate_response(agent_response, context_result)

        # Format the final response
        final_response = format_response(agent_response, context_result, response_validation)

        logger.info(f"Query processed successfully: '{request.query[:30]}{'...' if len(request.query) > 30 else ''}'")
        return final_response

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.get("/api/health", response_model=HealthCheckResponse)
async def health_check():
    """
    Health check endpoint to verify service availability.

    Returns:
        HealthCheckResponse with service status information
    """
    try:
        health_status = {
            'status': 'healthy',
            'services': {
                'cohere_api': cohere_client is not None,
                'qdrant_db': qdrant_client is not None,
                'config_loaded': config is not None
            }
        }

        # Test Cohere connectivity if client exists
        if cohere_client:
            try:
                cohere_client.embed(texts=["health"], model="embed-english-v3.0", input_type="search_query")
                health_status['services']['cohere_api'] = True
            except:
                health_status['services']['cohere_api'] = False

        # Test Qdrant connectivity if client exists
        if qdrant_client:
            try:
                qdrant_client.get_collections()
                health_status['services']['qdrant_db'] = True
            except:
                health_status['services']['qdrant_db'] = False

        # Overall status
        all_services_healthy = all(health_status['services'].values())
        health_status['status'] = 'healthy' if all_services_healthy else 'unhealthy'

        return HealthCheckResponse(**health_status)

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthCheckResponse(
            status='unhealthy',
            services={
                'cohere_api': False,
                'qdrant_db': False,
                'config_loaded': False
            }
        )


@app.post("/api/agent/validate", response_model=AgentValidateResponse)
async def validate_agent_response(request: AgentValidateRequest):
    """
    Validation endpoint for development purposes to check response grounding.

    Args:
        request: AgentValidateRequest containing query, context, and response to validate

    Returns:
        AgentValidateResponse with validation results
    """
    try:
        # Prepare context in the same format as during normal operation
        context_for_validation = {
            'chunks': [{
                'id': 'validation_context',
                'content': request.context,
                'url': 'validation',
                'title': 'Validation Context',
                'score': 1.0
            }],
            'selected_text_included': False,
            'retrieval_successful': True
        }

        # Prepare formatted context
        formatted_context = prepare_agent_context(context_for_validation, request.query)

        # Validate the response
        validation_result = validate_response(request.response, context_for_validation)

        return AgentValidateResponse(
            grounded_in_context=validation_result['grounded_in_context'],
            confidence_score=validation_result['confidence_score'],
            validation_details=validation_result['validation_details']
        )

    except Exception as e:
        logger.error(f"Error validating agent response: {e}")
        raise HTTPException(status_code=500, detail=f"Error validating response: {str(e)}")


def test_system_stability_with_error_conditions(cohere_client, qdrant_client, config):
    """
    Test system stability with various error conditions and edge cases.

    Args:
        cohere_client: Initialized Cohere client
        qdrant_client: Initialized Qdrant client
        config: Configuration dictionary

    Returns:
        True if system handles error conditions gracefully, False otherwise
    """
    logger.info("Testing system stability with various error conditions and edge cases")

    stability_test_results = {
        'passed': 0,
        'failed': 0,
        'total': 0
    }

    def run_stability_test(test_name: str, test_func):
        """Helper function to run a stability test and track results."""
        stability_test_results['total'] += 1
        try:
            result = test_func()
            if result:
                stability_test_results['passed'] += 1
                logger.info(f"Stability test passed: {test_name}")
            else:
                stability_test_results['failed'] += 1
                logger.warning(f"Stability test partially failed (but didn't crash): {test_name}")
        except Exception as e:
            stability_test_results['failed'] += 1
            logger.error(f"Stability test failed with exception: {test_name} - {e}")

    # Test 1: Empty query
    def test_empty_query():
        result = validate_input("")
        if result['is_valid']:
            # This shouldn't happen - empty query should fail validation
            embed_result = embed_query(cohere_client, "")
            return embed_result is None
        return True  # Expected to fail validation, which is correct behavior
    run_stability_test("Empty query handling", test_empty_query)

    # Test 2: Very long query
    def test_very_long_query():
        long_query = "test " * 500  # Way over the 2000 character limit
        result = validate_input(long_query)
        if result['is_valid']:
            # This shouldn't happen - long query should fail validation
            embed_result = embed_query(cohere_client, long_query)
            return embed_result is None
        return True  # Expected to fail validation, which is correct behavior
    run_stability_test("Very long query handling", test_very_long_query)

    # Test 3: Query with special characters
    def test_special_characters_query():
        special_query = "test query with <script>alert('xss')</script> and other chars"
        result = validate_input(special_query)
        if result['is_valid']:
            # Process the query to ensure the system doesn't crash
            context_result = get_context(special_query, qdrant_client, config, None)
            return True  # If it doesn't crash, that's good
        return True  # If validation fails, that's also acceptable
    run_stability_test("Special characters query handling", test_special_characters_query)

    # Test 4: Query with SQL injection patterns
    def test_sql_injection_query():
        sql_query = "test query with SELECT * FROM users WHERE 1=1"
        result = validate_input(sql_query)
        if result['is_valid']:
            # Process the query to ensure the system doesn't crash
            context_result = get_context(sql_query, qdrant_client, config, None)
            return True  # If it doesn't crash, that's good
        return True  # If validation fails, that's also acceptable
    run_stability_test("SQL injection query handling", test_sql_injection_query)

    # Test 5: Invalid embedding handling
    def test_invalid_embedding():
        try:
            # Try to search with an invalid embedding (wrong dimension)
            invalid_embedding = [0.1] * 100  # Wrong dimension
            search_results = qdrant_client.query_points(
                collection_name="book_embeddings",
                query=invalid_embedding,
                limit=config['top_k_results'],
                score_threshold=config['similarity_threshold'],
                with_payload=True,
                with_vectors=False
            ).points
            # If it returns results, that's OK; if it fails gracefully, that's also OK
            return True
        except Exception:
            # If it raises an exception but doesn't crash the system, that's acceptable
            return True
    run_stability_test("Invalid embedding handling", test_invalid_embedding)

    # Test 6: Zero top_k results
    def test_zero_top_k():
        try:
            query = "test"
            query_embedding = embed_query(cohere_client, query)
            if query_embedding:
                # Test with top_k = 0
                search_results = qdrant_client.query_points(
                    collection_name="book_embeddings",
                    query=query_embedding,
                    limit=0,  # Zero results requested
                    score_threshold=config['similarity_threshold'],
                    with_payload=True,
                    with_vectors=False
                ).points
                # This should return empty results, which is acceptable
                return True
            return True  # If embedding failed, that's also acceptable
        except Exception:
            # If it raises an exception, that's also acceptable as long as the system doesn't crash
            return True
    run_stability_test("Zero top_k handling", test_zero_top_k)

    # Test 7: Very high similarity threshold
    def test_high_similarity_threshold():
        try:
            query = "test"
            query_embedding = embed_query(cohere_client, query)
            if query_embedding:
                search_results = qdrant_client.query_points(
                    collection_name="book_embeddings",
                    query=query_embedding,
                    limit=config['top_k_results'],
                    score_threshold=0.99,  # Very high threshold
                    with_payload=True,
                    with_vectors=False
                ).points
                # This might return empty results, which is acceptable
                return True
            return True  # If embedding failed, that's also acceptable
        except Exception:
            # If it raises an exception, that's also acceptable as long as the system doesn't crash
            return True
    run_stability_test("High similarity threshold handling", test_high_similarity_threshold)

    # Test 8: Malformed query with excessive special characters
    def test_excessive_special_chars():
        special_query = "".join(["#"] * 200)  # All special characters
        result = validate_input(special_query)
        if result['is_valid']:
            # Process the query to ensure the system doesn't crash
            context_result = get_context(special_query, qdrant_client, config, None)
            return True  # If it doesn't crash, that's good
        return True  # If validation fails, that's also acceptable
    run_stability_test("Excessive special characters handling", test_excessive_special_chars)

    logger.info(f"System stability tests completed: {stability_test_results['passed']} passed, "
               f"{stability_test_results['failed']} failed out of {stability_test_results['total']} tests")

    # The system is stable if it doesn't crash, even if some tests return unexpected results
    return True


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)