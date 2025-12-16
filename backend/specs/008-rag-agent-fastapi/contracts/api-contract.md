# API Contract: Context-Aware RAG Agent with FastAPI

## Overview
This contract defines the expected behavior and interfaces for the FastAPI backend with OpenAI Agent SDK that answers queries using retrieved book content and optional user-selected text. The system prioritizes selected text in the agent context and strictly grounds responses in book content.

## Functional Endpoints

### POST /api/agent/query
**Purpose**: Process user query with optional selected text through the RAG agent
**Description**: Takes a user query and optional selected text, retrieves relevant context, processes through the agent, and returns a grounded response
**Parameters**: None (uses request body)
**Request Body**:
```
{
  "query": "string (required) - User's question or query",
  "selected_text": "string (optional) - Text selected by user in frontend",
  "user_preferences": {
    "context_priority": "string (optional) - How to prioritize context (default: selected_text_first)",
    "response_length": "string (optional) - Desired response length (default: medium)"
  }
}
```

**Response** (200 OK):
```
{
  "request_id": "string - Unique identifier for the request",
  "answer": "string - The agent's answer to the query",
  "sources": [
    {
      "content": "string - Snippet of the source content",
      "url": "string - Source URL",
      "title": "string - Source title",
      "similarity_score": "float - Relevance score (0.0-1.0)"
    }
  ],
  "confidence": "float - Confidence score for the response (0.0-1.0)",
  "grounded_in_context": "bool - Whether response is grounded in provided context",
  "response_time_ms": "int - Time taken to generate response",
  "fallback_used": "bool - Whether a fallback response was used"
}
```

**Error Response** (400, 422, 500):
```
{
  "detail": {
    "error_code": "string",
    "message": "Human-readable error message",
    "timestamp": "ISO 8601 datetime"
  }
}
```

### GET /api/health
**Purpose**: Check the health status of the RAG agent system
**Description**: Provides health check and service availability status
**Parameters**: None
**Response** (200 OK):
```
{
  "status": "string - Overall health status (healthy, degraded, error)",
  "timestamp": "ISO 8601 datetime - When the check was performed",
  "services": {
    "openai_api": "string - Status of OpenAI API connection (connected, disconnected, error)",
    "qdrant_db": "string - Status of Qdrant database connection (connected, disconnected, error)",
    "agent_service": "string - Status of agent service (operational, degraded, error)"
  }
}
```

### POST /api/agent/validate
**Purpose**: Validate agent reasoning and context grounding (for development/testing)
**Description**: Validates that a query and context would produce a properly grounded response
**Parameters**: None (uses request body)
**Request Body**:
```
{
  "query": "string (required) - Query to validate",
  "context": [
    {
      "content": "string - Context content",
      "source": "string - Source identifier",
      "priority": "string - Priority level (selected_text, high, medium, low)"
    }
  ],
  "expected_behavior": "string - Expected agent behavior"
}
```

**Response** (200 OK):
```
{
  "validation_id": "string - Unique identifier for the validation",
  "query": "string - Original query",
  "context_quality": "float - Quality score for provided context (0.0-1.0)",
  "grounding_validated": "bool - Whether response would be properly grounded",
  "validation_notes": "string - Additional validation information",
  "recommendations": "array[string] - Recommendations for improvement"
}
```

## Data Contracts

### AgentQueryRequest Object
```
{
  "query": "string (required) - User's question or query, max 2000 characters",
  "selected_text": "string (optional) - Text selected by user in frontend, max 5000 characters",
  "user_preferences": {
    "context_priority": "string (optional) - Context priority setting, default: 'selected_text_first'",
    "response_length": "string (optional) - Desired response length, default: 'medium'"
  }
}
```

### AgentQueryResponse Object
```
{
  "request_id": "string - Unique identifier for the request",
  "answer": "string - The agent's answer to the query",
  "sources": "array[SourceObject] - List of sources used to generate the answer",
  "confidence": "float (0.0-1.0) - Confidence score for the response",
  "grounded_in_context": "bool - Whether response is grounded in provided context",
  "response_time_ms": "int - Time taken to generate response",
  "fallback_used": "bool - Whether a fallback response was used"
}
```

### SourceObject
```
{
  "content": "string - Snippet of the source content",
  "url": "string - Source URL",
  "title": "string - Source title",
  "similarity_score": "float (0.0-1.0) - Relevance score"
}
```

### HealthCheckResponse Object
```
{
  "status": "string - Overall health status (healthy, degraded, error)",
  "timestamp": "ISO 8601 datetime - When the check was performed",
  "services": {
    "openai_api": "string - Status of OpenAI API connection",
    "qdrant_db": "string - Status of Qdrant database connection",
    "agent_service": "string - Status of agent service"
  }
}
```

## Error Responses
All endpoints follow standard FastAPI error response format:
- **400 Bad Request**: Invalid request parameters or body
- **422 Unprocessable Entity**: Request validation failed
- **500 Internal Server Error**: Server-side processing error
- **503 Service Unavailable**: External service (OpenAI, Qdrant) unavailable

## Configuration Requirements
The system expects these environment variables:
- `OPENAI_API_KEY`: API key for OpenAI services
- `QDRANT_URL`: URL for Qdrant Cloud instance
- `QDRANT_API_KEY`: API key for Qdrant Cloud
- `OPENAI_MODEL`: OpenAI model to use (default: gpt-4-turbo)
- `CONTEXT_WINDOW_LIMIT`: Maximum tokens for context (default: 120000)
- `SELECTED_TEXT_PRIORITY`: Priority weight for selected text (default: 0.8)
- `FALLBACK_MESSAGE`: Message when context is insufficient (default: "I cannot answer based on the provided context")
- `MAX_QUERY_LENGTH`: Maximum length of user query (default: 2000)
- `MAX_SELECTED_TEXT_LENGTH`: Maximum length of selected text (default: 5000)
- `API_RATE_LIMIT`: Rate limit for API requests (default: 60 per minute)

## Security Considerations
- All API keys stored in environment variables, never in code
- Input validation on all user-provided content
- Rate limiting to prevent abuse
- No persistent session state maintained
- Proper error handling that doesn't expose internal details