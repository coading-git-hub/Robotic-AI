# Research: Book Content Embeddings Ingestion Pipeline

## Decision Log

### Embedding Provider: Cohere
**Decision**: Use Cohere for generating embeddings
**Rationale**:
- Cost-effective with good performance for text embeddings
- Well-documented API with Python SDK
- Reliable service with good uptime
- Appropriate for book content processing

**Alternatives considered**:
- OpenAI: More expensive than Cohere
- Hugging Face: Self-hosted option but requires more infrastructure
- Google Vertex AI: Good but Cohere is simpler to integrate

### Vector Database: Qdrant Cloud Free Tier
**Decision**: Use Qdrant Cloud Free Tier for vector storage
**Rationale**:
- Managed vector search solution requiring minimal maintenance
- Good performance for semantic search
- Free tier sufficient for initial development
- Python client library available

**Alternatives considered**:
- Pinecone: Commercial option with good features but costs
- Weaviate: Open-source alternative but requires self-hosting
- Milvus: High-performance but more complex setup

### Chunking Strategy: Fixed-size with Overlap
**Decision**: Use 512-token chunks with 128-token overlap
**Rationale**:
- Provides consistent chunk sizes for processing
- Overlap helps maintain context across chunk boundaries
- Matches requirements from spec (FR-003)
- Balances between context preservation and processing efficiency

**Alternatives considered**:
- Sentence-based chunking: Less consistent sizes
- Paragraph-based: May be too large for embedding models
- Sliding window: More complex but similar to overlap approach

## Technical Implementation Research

### URL Fetching from Sitemap
- Use `requests` library to fetch sitemap.xml
- Parse with `xml.etree.ElementTree` or `beautifulsoup4`
- Extract all `<loc>` elements containing URLs
- Handle potential redirects and errors gracefully

### HTML Content Processing
- Use `beautifulsoup4` to parse HTML content
- Extract main content by identifying content containers
- Remove navigation, headers, footers, and other non-content elements
- Preserve text structure and formatting

### Text Chunking
- Use `langchain.text_splitter` or custom implementation
- Apply RecursiveCharacterTextSplitter for semantic chunking
- Configure chunk size (512 tokens) and overlap (128 tokens)
- Maintain chunk boundaries at sentence or paragraph breaks when possible

### Cohere Embedding Generation
- Use Cohere Python SDK (`cohere`)
- Configure API key via environment variable
- Handle rate limits and retry logic
- Batch requests for efficiency

### Qdrant Storage
- Use Qdrant Python client (`qdrant-client`)
- Create collection with appropriate vector dimensions
- Store metadata (URL, title, content, chunk index) with each vector
- Implement idempotency by using URL+chunk_index as unique identifiers

## Architecture Considerations

### Standalone Script Design
- Single Python script for simplicity
- Command-line interface for execution
- Configuration via environment variables
- Error handling and logging capabilities
- Progress tracking for large datasets

### Cloud Deployment Compatibility
- Use environment variables for all configuration
- Handle network failures gracefully
- Implement proper error logging
- Support for running in containerized environments