"""
Main FastAPI application for the unified API service.
This serves both RAG functionality and authentication endpoints on the same port.
"""
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from .database import create_tables, get_db
from .auth.routes import router as auth_routes  # Use custom auth routes instead of Better Auth
from .personalization import routes as personalization_routes
import os
import sys
import logging
from dotenv import load_dotenv

# Add the parent directory to the path so we can import rag_agent_api
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import required components for the RAG API
from rag_agent_api import (
    AgentQueryRequest,
    AgentQueryResponse,
    HealthCheckResponse,
    AgentValidateRequest,
    AgentValidateResponse,
    # Configuration and clients will be initialized globally
    load_configuration,
    initialize_cohere_client,
    initialize_qdrant_client,
    validate_input,
    get_context,
    prepare_agent_context,
    call_agent,
    validate_response,
    format_response,
    generate_fallback_response,
    # Global variables that need to be shared
    cohere_client as original_cohere_client,
    qdrant_client as original_qdrant_client,
    config as original_config
)

load_dotenv()

# Global variables for the unified app
cohere_client = None
qdrant_client = None
config = None

# Initialize configuration and clients
def ensure_initialized():
    """Ensure clients and configuration are initialized."""
    global cohere_client, qdrant_client, config

    if config is None:
        try:
            # Load configuration
            config = load_configuration()
            logging.info("Configuration loaded successfully")
        except Exception as e:
            logging.error(f"Failed to load configuration: {e}")
            raise

    if cohere_client is None:
        try:
            # Initialize Cohere client
            cohere_client = initialize_cohere_client(config['cohere_api_key'])
            if not cohere_client:
                logging.warning("Cohere client initialization failed, running in fallback mode")
        except Exception as e:
            logging.warning(f"Cohere client initialization error: {e}, running in fallback mode")

    if qdrant_client is None:
        try:
            # Initialize Qdrant client
            qdrant_client = initialize_qdrant_client(
                config['qdrant_url'],
                config['qdrant_api_key'],
                config['retrieval_timeout']
            )
            if not qdrant_client:
                logging.error("Failed to initialize Qdrant client")
                raise ValueError("Qdrant client initialization failed")
        except Exception as e:
            logging.error(f"Failed to initialize Qdrant client: {e}")
            raise

# Create tables on startup
create_tables()

# Create FastAPI app instance
app = FastAPI(
    title="Unified API Service",
    description="API for both RAG functionality and user authentication",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include auth routes (these match what the frontend expects: /api/auth/*)
app.include_router(auth_routes, prefix="/api/auth", tags=["Authentication"])

# Include personalization routes
app.include_router(personalization_routes.router, prefix="/api/personalization", tags=["Personalization"])

# Import and add RAG endpoints
from fastapi import APIRouter
rag_router = APIRouter()

# Add the RAG agent endpoints
@rag_router.post("/api/agent/query", response_model=AgentQueryResponse)
async def query_agent_endpoint(request: AgentQueryRequest):
    """
    Main endpoint to process user queries with optional selected text.
    """
    try:
        import logging as rag_logging
        logger = rag_logging.getLogger(__name__)

        logger.info(f"Processing query: '{request.query[:50]}{'...' if len(request.query) > 50 else ''}'")

        # Validate inputs
        validation_result = validate_input(request.query, request.selected_text)
        if not validation_result['is_valid']:
            raise HTTPException(status_code=400, detail=validation_result['error_message'])

        # Ensure clients are initialized
        try:
            ensure_initialized()
        except Exception as e:
            # Check if it's specifically a Qdrant issue (which is critical)
            # If it's just Cohere, we can continue with fallback
            if "Qdrant client initialization failed" in str(e):
                raise HTTPException(status_code=500, detail=f"Service initialization failed: {str(e)}")
            else:
                # Other initialization issues, but we can still proceed
                logger.warning(f"Non-critical initialization issue: {e}")

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

        # Check if Cohere client is available, if not use fallback
        global cohere_client
        if cohere_client:
            # Call the agent
            agent_response = call_agent(formatted_context, cohere_client, config)
        else:
            # Use fallback mechanism when Cohere API is unavailable
            logger.warning("Cohere client not available, using fallback response")
            agent_response = generate_fallback_response(
                validation_result['cleaned_query'],
                context_result['chunks']
            )

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

@rag_router.get("/api/health", response_model=HealthCheckResponse)
async def health_check_endpoint():
    """
    Health check endpoint to verify service availability.
    """
    try:
        try:
            ensure_initialized()
            health_status = {
                'status': 'healthy',
                'services': {
                    'cohere_api': True,
                    'qdrant_db': True,
                    'config_loaded': True
                }
            }

            # Test Cohere connectivity
            try:
                cohere_client.embed(texts=["health"], model="embed-english-v3.0", input_type="search_query")
                health_status['services']['cohere_api'] = True
            except Exception as e:
                logging.error(f"General error during Cohere health check: {e}")
                # Check if it's a rate limit error by looking at the error message or status code
                if hasattr(e, 'status_code') and e.status_code == 429:
                    logging.warning("Cohere rate limit reached during health check")
                elif "429" in str(e) or "rate limit" in str(e).lower() or "trial limit" in str(e).lower():
                    logging.warning("Cohere rate limit reached during health check")
                health_status['services']['cohere_api'] = False

            # Test Qdrant connectivity
            try:
                qdrant_client.get_collections()
                health_status['services']['qdrant_db'] = True
            except:
                health_status['services']['qdrant_db'] = False

        except Exception as e:
            health_status = {
                'status': 'unhealthy',
                'services': {
                    'cohere_api': False,
                    'qdrant_db': False,
                    'config_loaded': False
                }
            }

        # Overall status
        all_services_healthy = all(health_status['services'].values())
        health_status['status'] = 'healthy' if all_services_healthy else 'unhealthy'

        return HealthCheckResponse(**health_status)

    except Exception as e:
        logging.error(f"Health check failed: {e}")
        return HealthCheckResponse(
            status='unhealthy',
            services={
                'cohere_api': False,
                'qdrant_db': False,
                'config_loaded': False
            }
        )

@rag_router.post("/api/agent/validate", response_model=AgentValidateResponse)
async def validate_agent_response_endpoint(request: AgentValidateRequest):
    """
    Validation endpoint for development purposes to check response grounding.
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
        logging.error(f"Error validating agent response: {e}")
        raise HTTPException(status_code=500, detail=f"Error validating response: {str(e)}")

# Include the RAG router
app.include_router(rag_router)

@app.get("/")
def read_root():
    return {"message": "Unified API Service - RAG and Authentication"}

@app.get("/health")
def unified_health_check():
    return {"status": "healthy", "service": "unified-api"}

# Database dependency
def get_database():
    db = next(get_db())
    try:
        yield db
    finally:
        db.close()

if __name__ == "__main__":
    import uvicorn
    print("Starting Unified API Service on port 8002...")
    print("RAG API endpoints available at: http://localhost:8002/api/agent/*")
    print("Auth API endpoints available at: http://localhost:8002/api/auth/*")
    print("Health check available at: http://localhost:8002/api/health")
    uvicorn.run(app, host="0.0.0.0", port=8002, reload=False)