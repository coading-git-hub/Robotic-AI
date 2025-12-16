# API Contract: RAG Retrieval Validation System

## Overview
This contract defines the expected behavior and interfaces for the RAG retrieval validation system. The system validates that queries can be properly embedded, matched against stored vectors, and return relevant results with complete metadata.

## Functional Endpoints

### POST /validate-retrieval
**Purpose**: Validate the complete retrieval pipeline with a user query
**Description**: Takes a user query, embeds it, searches Qdrant, validates results, and returns validation report
**Parameters**:
  - `query` (string, required): The user query to validate
  - `top_k` (integer, optional): Number of results to retrieve (default: 5)
  - `threshold` (float, optional): Similarity threshold for relevance (default: 0.3)

**Request Body**:
```
{
  "query": "string (user query to validate)",
  "top_k": "integer (number of results, default 5)",
  "threshold": "float (similarity threshold, default 0.3)"
}
```

**Response**:
```
{
  "status": "success|error",
  "validation_results": {
    "query": "string (original query)",
    "query_embedding_success": boolean,
    "retrieval_success": boolean,
    "metadata_complete": boolean,
    "relevance_validated": boolean,
    "accuracy_score": "float (0.0-1.0)",
    "total_retrieved": integer,
    "relevant_count": integer,
    "execution_time_ms": integer
  },
  "retrieved_chunks": [
    {
      "content": "string (retrieved content chunk)",
      "url": "string (source URL)",
      "title": "string (page/section title)",
      "chunk_index": "integer",
      "similarity_score": "float (0.0-1.0)",
      "source_document": "string"
    }
  ]
}
```

### GET /health
**Purpose**: Check the health status of the retrieval validation system
**Description**: Provides health check and service availability status
**Parameters**: None
**Response**:
```
{
  "status": "healthy|degraded|error",
  "timestamp": "ISO 8601 datetime",
  "services": {
    "cohere_api": "connected|disconnected",
    "qdrant_db": "connected|disconnected",
    "retrieval_service": "operational|degraded"
  }
}
```

### POST /test-query
**Purpose**: Test a specific query against the retrieval system
**Description**: Validates a single query and returns detailed analysis
**Parameters**:
  - `query` (string, required): The query to test
  - `validate_relevance` (boolean, optional): Whether to validate relevance (default: true)

**Request Body**:
```
{
  "query": "string (query to test)",
  "validate_relevance": "boolean (validate relevance, default true)"
}
```

**Response**:
```
{
  "status": "success|error",
  "query_analysis": {
    "original_query": "string",
    "query_embedding": "array[float] (1024-dim vector)",
    "retrieval_results": {
      "chunks_retrieved": integer,
      "avg_similarity": "float",
      "top_similarity": "float",
      "metadata_completeness": "float (0.0-1.0)"
    },
    "relevance_validation": {
      "is_relevant": boolean,
      "confidence": "float (0.0-1.0)",
      "relevant_chunks": integer
    }
  }
}
```

## Data Contracts

### QueryVector Object
```
{
  "vector_id": "string (unique identifier)",
  "vector": "array[float] (1024-dim Cohere embedding)",
  "original_query": "string",
  "query_type": "string (factual|conceptual|etc.)",
  "created_at": "ISO 8601 datetime",
  "model_version": "string"
}
```

### RetrievedChunk Object
```
{
  "chunk_id": "string (content hash from storage)",
  "content": "string (text content)",
  "url": "string (source URL)",
  "title": "string (page/section title)",
  "chunk_index": "integer",
  "similarity_score": "float (0.0-1.0)",
  "source_document": "string",
  "retrieved_at": "ISO 8601 datetime"
}
```

### ValidationResult Object
```
{
  "validation_id": "string (unique identifier)",
  "query_vector_ref": "string (QueryVector ID)",
  "retrieved_chunks": "array[dict] (retrieved chunk details)",
  "total_retrieved": "integer",
  "relevant_count": "integer",
  "accuracy_score": "float (0.0-1.0)",
  "metadata_complete": "boolean",
  "validation_passed": "boolean",
  "validation_timestamp": "ISO 8601 datetime",
  "relevance_threshold": "float"
}
```

## Error Responses
All endpoints follow this error response format:
```
{
  "status": "error",
  "error_code": "string",
  "message": "Human-readable error message",
  "details": "Additional error details if available"
}
```

## Configuration Requirements
The system expects these environment variables:
- `COHERE_API_KEY`: API key for Cohere embedding service
- `QDRANT_URL`: URL for Qdrant Cloud instance
- `QDRANT_API_KEY`: API key for Qdrant Cloud
- `TOP_K_RESULTS`: Number of top results to retrieve (default: 5)
- `SIMILARITY_THRESHOLD`: Minimum similarity score for relevance (default: 0.3)
- `RETRIEVAL_TIMEOUT`: Timeout for retrieval operations in seconds (default: 10)