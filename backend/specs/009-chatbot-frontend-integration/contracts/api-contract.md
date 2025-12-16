# API Contract: RAG Chatbot Frontend Integration

## Overview
This contract defines the expected behavior and interfaces for the frontend-backend communication in the RAG chatbot system. The frontend sends user queries (with optional selected text) to the Agent SDK backend and receives formatted responses.

## Functional Endpoints

### POST /api/agent/query
**Purpose**: Process user query with optional selected text through the RAG agent
**Description**: Takes a user query and optional selected text, processes through the agent, and returns a grounded response
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
    "agent_sdk": "string - Status of Agent SDK connection (connected, disconnected, error)",
    "qdrant_db": "string - Status of Qdrant database connection (connected, disconnected, error)",
    "agent_service": "string - Status of agent service (operational, degraded, error)"
  }
}
```

## Frontend Component Interfaces

### ChatWidget Component Interface
```
Props:
  - isOpen: boolean (required) - Controls whether chat is visible
  - onClose: function (required) - Callback when chat is closed
  - backendUrl: string (required) - URL of the backend API
  - initialSelectedText: string (optional) - Pre-filled selected text

State:
  - messages: array of ChatMessage objects
  - inputValue: string
  - isLoading: boolean
  - error: string (nullable)

Methods:
  - sendMessage(text: string, selectedText?: string): Promise<void>
  - clearChat(): void
  - minimize(): void
```

### TextSelectionHandler Component Interface
```
Props:
  - onTextSelected: function (required) - Callback when text is selected
  - enabled: boolean (default: true) - Whether text selection is active

State:
  - selectedText: string
  - isTextSelected: boolean
  - selectionRange: Range object

Methods:
  - getSelectedText(): string
  - clearSelection(): void
```

### APIClient Service Interface
```
Props:
  - backendUrl: string (required) - Base URL for the backend API

Methods:
  - sendQuery(query: string, selectedText?: string): Promise<APIResponse>
  - checkHealth(): Promise<HealthResponse>
  - cancelRequest(requestId: string): void
```

## Data Contracts

### ChatMessage Object
```
{
  "id": "string - Unique message identifier",
  "content": "string - Message text content",
  "sender": "string - 'user' or 'agent'",
  "timestamp": "ISO 8601 datetime - When message was created",
  "type": "string - 'query', 'response', or 'system'",
  "sources": "array[SourceObject] - Sources referenced in agent response (optional)",
  "confidence": "float (0.0-1.0) - Confidence score for agent response (optional)"
}
```

### APIRequest Object
```
{
  "requestId": "string - Unique identifier for the API request",
  "query": "string - The user's query to the agent",
  "selectedText": "string (optional) - Text selected by user in the book",
  "timestamp": "ISO 8601 datetime - When the request was sent",
  "backendUrl": "string - URL of the FastAPI backend",
  "userPreferences": "object (optional) - User preferences for the request"
}
```

### APIResponse Object
```
{
  "responseId": "string - Unique identifier for the API response",
  "requestRef": "string - Reference to the original APIRequest",
  "response": "string - The agent's response to the query",
  "sources": "array[SourceObject] - List of sources used to generate the answer",
  "confidence": "float (0.0-1.0) - Confidence score for the response",
  "groundedInContext": "bool - Whether the response is grounded in provided context",
  "processingTimeMs": "int - Time taken to process the request in milliseconds",
  "timestamp": "ISO 8601 datetime - When the response was received",
  "fallbackUsed": "bool - Whether a fallback response was used"
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

## Error Responses
All endpoints follow standard HTTP error response format:
- **400 Bad Request**: Invalid request parameters or body
- **422 Unprocessable Entity**: Request validation failed
- **500 Internal Server Error**: Server-side processing error
- **503 Service Unavailable**: External service (OpenAI, Qdrant) unavailable

## Configuration Requirements
The frontend expects these configuration parameters:
- `BACKEND_API_URL`: URL of the FastAPI agent backend
- `CHAT_WIDGET_POSITION`: Position of floating widget (default: "bottom-right")
- `CHAT_WIDGET_SIZE`: Size of floating widget (default: "60px")
- `DEFAULT_FALLBACK_MESSAGE`: Message when context is insufficient
- `MAX_SELECTED_TEXT_LENGTH`: Maximum length of selected text (default: 2000)
- `REQUEST_TIMEOUT_MS`: Timeout for API requests (default: 30000)