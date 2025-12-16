# RAG Retrieval Pipeline Validation

This specification defines the implementation of a RAG retrieval pipeline validation system that queries Qdrant for relevant content using Cohere embeddings. The system validates retrieval pipeline functionality, ensures retrieved chunks match expected sections, verifies metadata integrity, and handles irrelevant queries gracefully.

## Overview

The system validates that:
1. User queries are properly embedded using the same Cohere model as content ingestion
2. Qdrant returns relevant content using cosine similarity search
3. Retrieved chunks contain complete metadata (URL, title, chunk_index)
4. Content is semantically relevant to user queries
5. Irrelevant queries are handled gracefully

## Key Components

- `embed_query(query)`: Generates Cohere embedding for user query
- `query_qdrant(query_vector, top_k)`: Queries Qdrant for top-k results
- `validate_retrieval(retrieved_chunks, query)`: Validates metadata completeness
- `validate_relevance(retrieved_chunks, query)`: Validates semantic relevance
- `handle_irrelevant_query(query)`: Handles queries with no relevant results
- `main(query)`: Orchestrates the complete validation pipeline

## Files in this Specification

- `spec.md`: Feature requirements and user scenarios
- `plan.md`: Implementation architecture and design
- `research.md`: Technical research and decision documentation
- `data-model.md`: Data structure definitions
- `contracts/api-contract.md`: API interface specifications
- `quickstart.md`: Setup and usage instructions
- `checklists/`: Implementation checklists

## Technical Decisions

- **Embeddings**: Cohere `embed-english-v3.0` model (same as ingestion)
- **Retrieval**: Vector-only with cosine similarity
- **Results**: Top-k (default: 5) with configurable threshold (default: 0.3)
- **Validation**: Multi-level validation for accuracy and relevance
- **Configuration**: Environment variables via python-dotenv