# Quickstart Guide: RAG Retrieval Pipeline Validation

**Feature**: 007-rag-retrieval-validation
**Created**: 2025-12-13

## Overview

This guide provides quick setup instructions for the RAG retrieval pipeline validation system. The system queries Qdrant for relevant content using Cohere embeddings and validates that retrieved chunks match expected sections with intact metadata.

## Prerequisites

- Python 3.9 or higher
- pip package manager
- Access to Cohere API (API key)
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
pip install cohere qdrant-client python-dotenv
```

### 4. Configure Environment Variables
Create a `.env` file with the following content:

```env
COHERE_API_KEY=your_cohere_api_key_here
QDRANT_URL=your_qdrant_cloud_url_here
QDRANT_API_KEY=your_qdrant_api_key_here
TOP_K_RESULTS=5
SIMILARITY_THRESHOLD=0.3
RETRIEVAL_TIMEOUT=10
```

### 5. Prepare the retrieval.py File
Create `retrieval.py` with the implementation containing:
- `embed_query(query)` - Function to generate Cohere embedding for user query
- `query_qdrant(query_vector, top_k=5)` - Function to query Qdrant for top-k results
- `validate_retrieval(retrieved_chunks, query)` - Function to validate metadata completeness
- `validate_relevance(retrieved_chunks, query)` - Function to validate semantic relevance
- `handle_irrelevant_query(query)` - Function to handle queries with no relevant results
- `main(query)` - Orchestrator function for the complete validation pipeline

## Running Validation

### Execute a Single Query Validation
```bash
python retrieval.py --query "your query here"
```

### Run Batch Validation
```bash
python retrieval.py --batch-validation
```

### Test with Sample Queries
```bash
python retrieval.py --test-queries
```

## Configuration Options

- `TOP_K_RESULTS`: Number of top results to retrieve (default: 5)
- `SIMILARITY_THRESHOLD`: Minimum similarity score for relevance (default: 0.3)
- `RETRIEVAL_TIMEOUT`: Timeout for retrieval operations in seconds (default: 10)

## Validation Output

The system will output validation results including:
1. Query embedding success rate
2. Retrieval success rate
3. Metadata completeness validation
4. Relevance validation scores
5. Performance metrics (latency, throughput)
6. Error handling validation

## Troubleshooting

### Common Issues

1. **API Rate Limits**: If hitting Cohere rate limits, the system implements exponential backoff
2. **Network Issues**: The system has built-in retry logic for transient failures
3. **No Results**: Queries unrelated to book content will return low-confidence results
4. **Invalid Configuration**: Verify all environment variables are set correctly

### Validation Commands

Check system health:
```bash
python retrieval.py --health-check
```

Run comprehensive validation:
```bash
python retrieval.py --full-validation
```

## Next Steps

- Monitor retrieval accuracy and relevance metrics
- Fine-tune similarity thresholds based on validation results
- Add more test queries to improve validation coverage
- Integrate validation into CI/CD pipeline for continuous monitoring