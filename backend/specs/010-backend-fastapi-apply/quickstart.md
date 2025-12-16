# Quickstart Guide: Backend FastAPI Implementation

## Overview

The backend FastAPI implementation is already complete and ready to use. It provides a RAG (Retrieval-Augmented Generation) agent API with comprehensive functionality.

## Prerequisites

- Python 3.8+
- pip package manager

## Setup Instructions

### 1. Navigate to Backend Directory
```bash
cd backend
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Ensure your `.env` file contains the required API keys and configuration:

```env
QDRANT_URL="your_qdrant_url"
QDRANT_API_KEY="your_qdrant_api_key"
COHERE_API_KEY="your_cohere_api_key"
```

### 4. Start the FastAPI Server
```bash
python rag_agent_api.py
```

The API will be available at `http://localhost:8000`

## API Endpoints

### Query Endpoint
- **POST** `/api/agent/query`
- Query the RAG agent with optional selected text
- Example request:
```json
{
  "query": "Your question here",
  "selected_text": "Optional text to prioritize in context"
}
```

### Health Check
- **GET** `/api/health`
- Check service availability

### Response Validation
- **POST** `/api/agent/validate`
- Validate if a response is grounded in context

### Documentation
- **GET** `/docs` - Swagger UI
- **GET** `/redoc` - ReDoc interface

## Verification

To verify the API is working:
1. Visit `http://localhost:8000/docs` to see the interactive API documentation
2. Test the health endpoint: `GET http://localhost:8000/api/health`
3. The API should return status information about connected services

## Next Steps

- Integrate with your frontend application
- Configure your production environment variables
- Set up monitoring and logging as needed