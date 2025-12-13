# API Documentation for Physical AI & Humanoid Robotics RAG System

## Overview

The Physical AI & Humanoid Robotics RAG API provides endpoints for:
- User authentication and management
- Chat interactions with the RAG system
- Content search and retrieval
- Progress tracking
- Health monitoring

## Authentication

The API uses JWT (JSON Web Tokens) for authentication. After successful login, users receive both access and refresh tokens.

### Token Refresh
- Endpoint: `POST /auth/refresh`
- Include the refresh token in the request body
- Returns new access and refresh tokens

## Available Endpoints

### Authentication (`/auth`)
- `POST /auth/register` - Register a new user
- `POST /auth/login` - Authenticate and get tokens
- `POST /auth/refresh` - Refresh access token
- `POST /auth/logout` - Logout (client-side token invalidation)
- `GET /auth/me` - Get current user profile

### Chat & RAG (`/chat`)
- `POST /chat/query` - Submit a query to the RAG system
- `GET /chat/session/{session_id}` - Get chat session history
- `POST /chat/session` - Create a new chat session
- `POST /chat/feedback` - Submit feedback for a query response

### Search (`/search`)
- `POST /search` - Search book content with semantic search
- `GET /content/{content_id}` - Get specific content item

### Progress Tracking (`/progress`)
- `GET /progress/user/{user_id}/overview` - Get user progress overview
- `POST /progress/record` - Record user progress for content

### Health (`/health`)
- `GET /health` - Basic health check
- `GET /health/status` - Detailed system status

## Rate Limiting

The API implements rate limiting to prevent abuse:
- Authenticated users: 100 requests per minute
- Anonymous users: 10 requests per minute
- Query endpoints: 10 queries per minute per user

## Error Handling

The API returns structured error responses:

```json
{
  "detail": {
    "error": {
      "code": "ERROR_CODE",
      "message": "Error description",
      "details": {}
    }
  }
}
```

## Response Format

Successful responses follow this format:

```json
{
  "success": true,
  "data": {},
  "message": "Success message",
  "timestamp": "2025-12-12T10:30:00Z",
  "request_id": "unique-request-id"
}
```

## Common Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| AUTH_001 | 401 | Invalid credentials |
| AUTH_002 | 401 | Token expired |
| AUTH_003 | 403 | Insufficient permissions |
| VALIDATION_001 | 400 | Invalid request format |
| VALIDATION_002 | 400 | Missing required fields |
| NOT_FOUND_001 | 404 | Resource not found |
| RATE_LIMIT_001 | 429 | Rate limit exceeded |
| SYSTEM_001 | 500 | Internal server error |

## Security Considerations

- All sensitive data is encrypted in transit using HTTPS
- JWT tokens are used for authentication with short expiration times
- Rate limiting prevents abuse and ensures fair usage
- Input validation prevents injection attacks
- Content filtering ensures responses are based only on book content