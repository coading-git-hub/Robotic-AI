# ADR-002: RAG Retrieval Architecture

**Date**: 2025-12-13
**Status**: Accepted
**Authors**: Claude Code

## Context

For the RAG retrieval validation system, we need to make key architectural decisions about:
1. How to query Qdrant for relevant content using Cohere embeddings
2. How to validate that retrieved chunks match expected sections
3. How to ensure metadata integrity during retrieval
4. How to handle irrelevant queries gracefully

The system needs to validate the end-to-end retrieval pipeline, ensuring that user queries result in semantically relevant content being returned with complete metadata, while handling edge cases appropriately.

## Decision

### Retrieval Method: Vector-Only with Cosine Similarity

We will use vector-only retrieval with cosine similarity for semantic matching:
- **Method**: Cosine similarity search in Qdrant
- **Model**: Cohere embed-english-v3.0 (same as ingestion pipeline)
- **Input Type**: search_query for queries, matching search_document for stored content
- **Distance**: Cosine similarity in Qdrant

### Result Configuration: Top-K with Threshold Filtering

We will retrieve top-k results with configurable similarity threshold:
- **Top-K**: 5 results by default (configurable)
- **Threshold**: 0.3 minimum similarity score (configurable)
- **Metadata**: All results include complete metadata (URL, title, chunk_index)

### Validation Strategy: Multi-Level Validation

We will implement comprehensive validation at multiple levels:
- **Metadata Validation**: Verify complete metadata for all retrieved chunks
- **Relevance Validation**: Assess semantic relevance using similarity scores
- **Content Matching**: Validate that content addresses query topic
- **Error Handling**: Graceful handling of irrelevant queries

### Architecture: Modular Python Functions

We will implement the system as modular Python functions:
- `embed_query()` - Generate Cohere embedding for user query
- `query_qdrant()` - Query Qdrant for top-k results
- `validate_retrieval()` - Validate metadata completeness
- `validate_relevance()` - Validate semantic relevance
- `handle_irrelevant_query()` - Handle queries with no relevant results
- `main()` - Orchestrate the complete validation pipeline

## Rationale

### Vector-Only Retrieval with Cosine Similarity
- **Performance**: Cosine similarity is efficient for high-dimensional embeddings
- **Accuracy**: Works well with Cohere embeddings for semantic matching
- **Consistency**: Uses same embedding space as ingestion pipeline
- **Standard**: Industry standard for semantic search applications

### Top-K with Threshold Configuration
- **Usability**: 5 results provide sufficient context without overwhelming users
- **Flexibility**: Configurable parameters allow tuning for different use cases
- **Quality**: Threshold filtering ensures minimum relevance standards
- **Efficiency**: Limited results maintain good performance

### Multi-Level Validation
- **Comprehensiveness**: Multiple validation layers ensure quality
- **Transparency**: Clear validation results for debugging and monitoring
- **Reliability**: Redundant checks prevent false positives
- **Measurability**: Quantifiable validation metrics

### Modular Architecture
- **Maintainability**: Clear separation of concerns
- **Testability**: Each function can be tested independently
- **Flexibility**: Easy to modify individual components
- **Reusability**: Functions can be used in other contexts

## Alternatives Considered

### Retrieval Methods
1. **Hybrid Search** (keyword + vector) - More complex, unnecessary for this use case
2. **Euclidean Distance** - Less effective for high-dimensional embeddings
3. **Dot Product Similarity** - Not normalized, harder to set consistent thresholds

### Result Configuration
1. **Fixed Results** (no configuration) - Less flexible for different scenarios
2. **Dynamic Results** (AI-determined) - More complex, less predictable
3. **Binary Results** (relevant/not) - Less nuanced than graded relevance

### Validation Approaches
1. **Single Validation** - Less comprehensive coverage
2. **External Validation** - More complex, less integrated
3. **No Validation** - Would not meet requirements

## Consequences

### Positive
- Cohere embeddings provide high-quality vectors for search tasks
- Cosine similarity ensures consistent relevance scoring
- Configurable parameters allow optimization for different scenarios
- Comprehensive validation ensures high-quality results
- Modular design enables easy maintenance and testing

### Negative
- Dependency on external APIs (Cohere, Qdrant) introduces potential failure points
- Fixed top-k may not be optimal for all query types
- Multiple validation layers add computational overhead
- Configuration complexity requires more setup

## Implementation

The implementation will be in `retrieval.py` with the following characteristics:
- Environment variable configuration for all service credentials
- Comprehensive error handling and retry logic
- Logging at each stage of the pipeline
- Validation functions to ensure retrieval accuracy
- Consistent embedding model with ingestion pipeline