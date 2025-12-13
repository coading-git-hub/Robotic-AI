# API Contracts: RAG System for Physical AI & Humanoid Robotics Book

**Feature**: course-physical-ai-humanoid
**Created**: 2025-12-12
**Status**: Complete
**Plan**: [implementation-plan.md](../architecture/implementation-plan.md)

## Overview

This document defines the API contracts for the RAG (Retrieval-Augmented Generation) system that powers the embedded chatbot functionality in the Physical AI & Humanoid Robotics book. The system is built on FastAPI and follows RESTful principles with JSON responses.

## Base API Configuration

### Base URL
```
https://api.physical-ai-book.com/v1
```

### Common Headers
```
Content-Type: application/json
Accept: application/json
Authorization: Bearer <token> (for authenticated endpoints)
X-Session-ID: <session-id> (for maintaining conversation context)
```

### Common Response Format
```json
{
  "success": true,
  "data": {},
  "message": "Success message",
  "timestamp": "2025-12-12T10:30:00Z",
  "request_id": "unique-request-id"
}
```

### Error Response Format
```json
{
  "success": false,
  "error": {
    "code": "ERROR_CODE",
    "message": "Error description",
    "details": {}
  },
  "timestamp": "2025-12-12T10:30:00Z",
  "request_id": "unique-request-id"
}
```

## Authentication Endpoints

### POST /auth/login
Authenticate user and get access token

**Request:**
```json
{
  "email": "user@example.com",
  "password": "user_password"
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "access_token": "jwt_token",
    "refresh_token": "refresh_token",
    "user": {
      "id": "user-uuid",
      "username": "username",
      "email": "user@example.com",
      "role": "student"
    }
  },
  "message": "Authentication successful",
  "timestamp": "2025-12-12T10:30:00Z",
  "request_id": "unique-request-id"
}
```

### POST /auth/refresh
Refresh access token

**Request:**
```json
{
  "refresh_token": "refresh_token"
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "access_token": "new_jwt_token"
  },
  "message": "Token refreshed",
  "timestamp": "2025-12-12T10:30:00Z",
  "request_id": "unique-request-id"
}
```

## RAG Chatbot Endpoints

### POST /chat/query
Submit a query to the RAG system

**Request:**
```json
{
  "query": "How does ROS 2 differ from ROS 1?",
  "selected_text": "ROS 2 is the next generation of the Robot Operating System...",
  "context_window": 5,
  "max_tokens": 500,
  "temperature": 0.7,
  "user_id": "user-uuid", // Optional
  "session_id": "session-uuid" // Optional
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "response": "ROS 2 differs from ROS 1 in several key ways...",
    "confidence_score": 0.85,
    "retrieved_chunks": [
      {
        "id": "chunk-uuid",
        "content": "The main differences include...",
        "source": "lesson-uuid",
        "similarity_score": 0.92
      }
    ],
    "query_id": "query-uuid",
    "session_id": "session-uuid",
    "response_time_ms": 450
  },
  "message": "Query processed successfully",
  "timestamp": "2025-12-12T10:30:00Z",
  "request_id": "unique-request-id"
}
```

### GET /chat/session/{session_id}
Get chat session history

**Response:**
```json
{
  "success": true,
  "data": {
    "session": {
      "id": "session-uuid",
      "title": "ROS 2 vs ROS 1 Discussion",
      "active": true,
      "created_at": "2025-12-12T10:00:00Z",
      "last_activity_at": "2025-12-12T10:30:00Z",
      "user_id": "user-uuid"
    },
    "messages": [
      {
        "id": "message-uuid",
        "type": "query",
        "content": "How does ROS 2 differ from ROS 1?",
        "timestamp": "2025-12-12T10:05:00Z"
      },
      {
        "id": "message-uuid",
        "type": "response",
        "content": "ROS 2 differs from ROS 1 in several key ways...",
        "confidence_score": 0.85,
        "timestamp": "2025-12-12T10:05:01Z"
      }
    ]
  },
  "message": "Session retrieved successfully",
  "timestamp": "2025-12-12T10:30:00Z",
  "request_id": "unique-request-id"
}
```

### POST /chat/session
Create a new chat session

**Request:**
```json
{
  "title": "New Chat Session",
  "user_id": "user-uuid" // Optional
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "session": {
      "id": "session-uuid",
      "title": "New Chat Session",
      "active": true,
      "created_at": "2025-12-12T10:30:00Z",
      "last_activity_at": "2025-12-12T10:30:00Z",
      "user_id": "user-uuid"
    }
  },
  "message": "Session created successfully",
  "timestamp": "2025-12-12T10:30:00Z",
  "request_id": "unique-request-id"
}
```

### POST /chat/feedback
Submit feedback for a query response

**Request:**
```json
{
  "query_id": "query-uuid",
  "rating": 5, // 1-5 scale
  "comment": "The response was very helpful and accurate."
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "feedback_id": "feedback-uuid",
    "query_id": "query-uuid",
    "rating": 5
  },
  "message": "Feedback submitted successfully",
  "timestamp": "2025-12-12T10:30:00Z",
  "request_id": "unique-request-id"
}
```

## Content Search Endpoints

### POST /search
Search book content with semantic search

**Request:**
```json
{
  "query": "navigation in ROS 2",
  "filters": {
    "module_id": "module-uuid", // Optional
    "lesson_id": "lesson-uuid", // Optional
    "content_type": ["theory", "hands_on"], // Optional
    "difficulty_level": ["beginner", "intermediate"] // Optional
  },
  "limit": 10,
  "offset": 0
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "results": [
      {
        "id": "content-uuid",
        "title": "ROS 2 Navigation System",
        "content_type": "lesson",
        "module_id": "module-uuid",
        "module_title": "ROS 2 Fundamentals",
        "content_preview": "The navigation system in ROS 2...",
        "similarity_score": 0.92,
        "url": "/modules/ros2/navigation"
      }
    ],
    "total_count": 25,
    "query_id": "search-query-uuid"
  },
  "message": "Search completed successfully",
  "timestamp": "2025-12-12T10:30:00Z",
  "request_id": "unique-request-id"
}
```

### GET /content/{content_id}
Get specific content item

**Response:**
```json
{
  "success": true,
  "data": {
    "content": {
      "id": "content-uuid",
      "title": "ROS 2 Navigation System",
      "content_type": "lesson",
      "module_id": "module-uuid",
      "module_title": "ROS 2 Fundamentals",
      "content": "# ROS 2 Navigation System\n\nThe navigation system...",
      "difficulty_level": "intermediate",
      "estimated_reading_time": 15,
      "created_at": "2025-12-12T10:00:00Z",
      "updated_at": "2025-12-12T10:00:00Z"
    }
  },
  "message": "Content retrieved successfully",
  "timestamp": "2025-12-12T10:30:00Z",
  "request_id": "unique-request-id"
}
```

## User Progress Endpoints

### GET /progress/user/{user_id}/overview
Get user progress overview

**Response:**
```json
{
  "success": true,
  "data": {
    "user": {
      "id": "user-uuid",
      "username": "username",
      "email": "user@example.com"
    },
    "progress_summary": {
      "total_modules": 13,
      "completed_modules": 3,
      "in_progress_modules": 1,
      "completion_percentage": 0.31,
      "total_time_spent": 420, // minutes
      "current_module": "module-uuid",
      "current_lesson": "lesson-uuid"
    },
    "module_progress": [
      {
        "module_id": "module-uuid",
        "module_title": "Foundations of Physical AI",
        "completion_percentage": 1.00,
        "status": "completed",
        "time_spent": 180
      }
    ]
  },
  "message": "Progress overview retrieved successfully",
  "timestamp": "2025-12-12T10:30:00Z",
  "request_id": "unique-request-id"
}
```

### POST /progress/record
Record user progress for content

**Request:**
```json
{
  "user_id": "user-uuid",
  "content_id": "lesson-uuid",
  "content_type": "lesson",
  "status": "completed",
  "completion_percentage": 1.00,
  "time_spent": 45, // minutes
  "score": 0.85 // Optional, for assessments
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "progress_record": {
      "id": "progress-uuid",
      "user_id": "user-uuid",
      "content_id": "lesson-uuid",
      "content_type": "lesson",
      "status": "completed",
      "completion_percentage": 1.00,
      "time_spent": 45,
      "score": 0.85,
      "created_at": "2025-12-12T10:30:00Z",
      "updated_at": "2025-12-12T10:30:00Z"
    }
  },
  "message": "Progress recorded successfully",
  "timestamp": "2025-12-12T10:30:00Z",
  "request_id": "unique-request-id"
}
```

## System Health Endpoints

### GET /health
Check system health

**Response:**
```json
{
  "success": true,
  "data": {
    "status": "healthy",
    "timestamp": "2025-12-12T10:30:00Z",
    "services": {
      "api": "healthy",
      "database": "healthy",
      "vector_store": "healthy",
      "llm_service": "healthy"
    }
  },
  "message": "System is healthy",
  "timestamp": "2025-12-12T10:30:00Z",
  "request_id": "unique-request-id"
}
```

### GET /health/metrics
Get system metrics

**Response:**
```json
{
  "success": true,
  "data": {
    "metrics": {
      "total_queries": 1250,
      "avg_response_time": 450, // ms
      "active_sessions": 45,
      "content_chunks": 2500,
      "daily_active_users": 120,
      "query_success_rate": 0.98
    },
    "timestamp": "2025-12-12T10:30:00Z"
  },
  "message": "Metrics retrieved successfully",
  "timestamp": "2025-12-12T10:30:00Z",
  "request_id": "unique-request-id"
}
```

## Error Codes

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
| RAG_001 | 500 | RAG system error |
| RAG_002 | 422 | Query too complex |
| CONTENT_001 | 404 | Content not found |
| PROGRESS_001 | 400 | Invalid progress data |

## Rate Limits

- **Authenticated users**: 100 requests per minute
- **Anonymous users**: 10 requests per minute
- **Query endpoints**: 10 queries per minute per user
- **Session creation**: 5 sessions per hour per user

## Security Considerations

- All sensitive data is encrypted in transit using HTTPS
- JWT tokens are used for authentication with short expiration times
- Rate limiting prevents abuse and ensures fair usage
- Input validation prevents injection attacks
- Content filtering ensures responses are based only on book content
- Session management prevents unauthorized access to conversation history