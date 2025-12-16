# Quickstart Guide: Context-Aware RAG Agent with FastAPI

**Feature**: 008-rag-agent-fastapi
**Created**: 2025-12-13

## Overview

This guide provides quick setup instructions for the FastAPI backend with OpenAI Agent SDK that answers queries using retrieved book content and optional user-selected text. The system prioritizes selected text in the agent context and strictly grounds responses in book content.

## Prerequisites

- Python 3.9 or higher
- pip package manager
- Access to OpenAI API (API key)
- Access to Qdrant Cloud (API key and URL)
- Previously ingested content in Qdrant from the book ingestion pipeline

## Setup Instructions

### 1. Clone the Repository
```bash
git clone <repository-url>
cd <repository-directory>
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install fastapi openai python-dotenv uvicorn pydantic requests qdrant-client
```

### 4. Configure Environment Variables
Create a `.env` file with the following content:

```env
OPENAI_API_KEY=your_openai_api_key_here
QDRANT_URL=your_qdrant_cloud_url_here
QDRANT_API_KEY=your_qdrant_api_key_here
OPENAI_MODEL=gpt-4-turbo
CONTEXT_WINDOW_LIMIT=120000
SELECTED_TEXT_PRIORITY=0.8
FALLBACK_MESSAGE=I cannot answer based on the provided context
MAX_QUERY_LENGTH=2000
MAX_SELECTED_TEXT_LENGTH=5000
API_RATE_LIMIT=60
```

### 5. Prepare the main.py File
Create `main.py` with the FastAPI application containing:
- Input validation functions
- Context retrieval from Qdrant
- Agent context preparation with prioritization
- OpenAI Agent SDK integration
- Response validation and formatting
- FastAPI endpoints for query processing

## Running the Service

### Start the Development Server
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Start the Production Server
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

## API Usage

### Query the Agent
```bash
curl -X POST http://localhost:8000/api/agent/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is ROS 2?",
    "selected_text": "ROS 2 (Robot Operating System 2) is a set of software libraries and tools that help you build robot applications."
  }'
```

### Health Check
```bash
curl -X GET http://localhost:8000/api/health
```

### Validate Agent Reasoning (Development)
```bash
curl -X POST http://localhost:8000/api/agent/validate \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Explain ROS 2 concepts",
    "context": [
      {
        "content": "ROS 2 provides libraries and tools for building robot applications",
        "source": "book/chapter1",
        "priority": "high"
      }
    ]
  }'
```

## Configuration Options

- `OPENAI_MODEL`: OpenAI model to use (default: gpt-4-turbo)
- `CONTEXT_WINDOW_LIMIT`: Maximum tokens for context (default: 120000)
- `SELECTED_TEXT_PRIORITY`: Priority weight for selected text (default: 0.8)
- `FALLBACK_MESSAGE`: Message when context is insufficient
- `MAX_QUERY_LENGTH`: Maximum length of user query (default: 2000)
- `MAX_SELECTED_TEXT_LENGTH`: Maximum length of selected text (default: 5000)
- `API_RATE_LIMIT`: Rate limit for API requests (default: 60 per minute)

## Response Format

The API returns responses in the following format:
1. `answer`: The agent's response to the query
2. `sources`: List of sources used to generate the answer
3. `confidence`: Confidence score for the response (0.0-1.0)
4. `grounded_in_context`: Whether the response is grounded in provided context
5. `response_time_ms`: Time taken to generate the response
6. `fallback_used`: Whether a fallback response was used

## Troubleshooting

### Common Issues

1. **API Rate Limits**: If hitting OpenAI rate limits, the system implements exponential backoff
2. **Network Issues**: The system has built-in retry logic for transient failures
3. **Context Too Long**: If combined context exceeds token limits, it's intelligently truncated
4. **Invalid Configuration**: Verify all environment variables are set correctly

### Health Check Commands

Check service health:
```bash
curl http://localhost:8000/api/health
```

Monitor service logs:
```bash
tail -f logs/app.log
```

## Next Steps

- Integrate with the frontend application
- Monitor agent response quality and grounding
- Fine-tune context prioritization based on usage
- Add monitoring and alerting for production deployment