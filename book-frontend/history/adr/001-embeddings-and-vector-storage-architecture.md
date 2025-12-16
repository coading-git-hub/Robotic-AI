# ADR-001: Embeddings and Vector Storage Architecture

**Date**: 2025-12-13
**Status**: Accepted
**Authors**: Claude Code

## Context

For the book content embeddings ingestion pipeline, we need to make key architectural decisions about:
1. Which embedding model to use for generating vector representations
2. Which vector database to use for storage and retrieval
3. How to structure the ingestion pipeline for optimal performance and reliability

The system needs to fetch content from deployed book URLs, process it (clean and chunk), generate embeddings, and store them in a vector database with metadata. This will enable the RAG system to provide accurate responses based on book content.

## Decision

### Embedding Model: Cohere embed-english-v3.0

We will use Cohere's `embed-english-v3.0` model for generating embeddings with the following parameters:
- **Model**: embed-english-v3.0
- **Input Type**: search_document (appropriate for document search)
- **Dimensions**: 1024

### Vector Database: Qdrant Cloud

We will use Qdrant Cloud as the vector database with the following configuration:
- **Storage**: Qdrant Cloud (Free Tier)
- **Collection**: book_embeddings
- **Vector Size**: 1024 dimensions (matching Cohere output)
- **Distance**: Cosine similarity
- **Deduplication**: Content hash-based (SHA-256) as vector ID

### Ingestion Pipeline Architecture: Single Python Script

We will implement the ingestion pipeline as a single Python script (`main.py`) with modular functions:
- `get_all_urls()` - Fetches URLs from sitemap.xml
- `extract_text_from_url()` - Extracts clean text using BeautifulSoup
- `chunk_text()` - Splits content using RecursiveCharacterTextSplitter
- `generate_embeddings()` - Creates embeddings with Cohere API
- `store_in_qdrant()` - Stores vectors with metadata
- `main()` - Orchestrates the complete pipeline

## Rationale

### Cohere embed-english-v3.0
- **Performance**: Optimized for English technical documentation which matches our book content
- **Quality**: High performance for retrieval tasks with consistent quality metrics
- **Cost**: Good cost-to-quality ratio with free tier available
- **Reliability**: Stable API with good documentation and support
- **Features**: Supports batch processing to optimize API usage

### Qdrant Cloud
- **Scalability**: Cloud-based solution that can scale with content volume
- **Performance**: Optimized for similarity search with cosine distance
- **Features**: Rich filtering capabilities and metadata storage
- **Ease of Use**: Managed service with minimal operational overhead
- **Free Tier**: Sufficient for initial implementation and testing
- **Deduplication**: Content hash as ID prevents duplicate storage

### Single Python Script Architecture
- **Simplicity**: Easy to deploy and maintain as a single file
- **Cloud Compatibility**: Can run in cloud environments with environment variable configuration
- **Modularity**: Functions are modular with clear separation of concerns
- **Testability**: Each function can be tested independently
- **Reproducibility**: Single file approach ensures consistent deployments

## Alternatives Considered

### Embedding Models
1. **OpenAI embeddings** - More expensive, would require different pricing model
2. **Sentence Transformers (self-hosted)** - More complex setup, adds operational overhead
3. **Cohere embed-multilingual-v3.0** - More expensive, not needed for English-only content

### Vector Databases
1. **Pinecone** - More expensive, less familiar with team
2. **Weaviate Cloud** - More complex setup, similar functionality to Qdrant
3. **Self-hosted PostgreSQL with pgvector** - More operational overhead, less optimized for similarity search

### Architecture Approaches
1. **Microservices** - More complex for this use case, unnecessary overhead
2. **Multiple Python files** - Would add complexity without significant benefit
3. **Serverless functions** - More complex orchestration, potentially higher cost

## Consequences

### Positive
- Cohere embeddings provide high-quality vectors for search tasks
- Qdrant Cloud offers managed service with good performance
- Single script architecture simplifies deployment and maintenance
- Content hash-based deduplication ensures no duplicates on re-runs
- Cloud-compatible configuration via environment variables

### Negative
- Dependency on external APIs (Cohere, Qdrant) introduces potential failure points
- Free tier limitations may require migration to paid plans as content grows
- Single script approach may become unwieldy if functionality expands significantly
- Rate limits on Cohere API may require careful batch processing

## Implementation

The implementation will be in `main.py` with the following characteristics:
- Environment variable configuration for all service credentials
- Comprehensive error handling and retry logic
- Logging at each stage of the pipeline
- Validation functions to ensure data integrity
- Idempotent operation to prevent duplicates on re-runs