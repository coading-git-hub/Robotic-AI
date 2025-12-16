---
id: 3-embeddings-architecture-adr
title: Embeddings Architecture Decision Record
stage: adr
date_iso: 2025-12-13
surface: agent
model: Claude Sonnet 4.5
feature: 006-book-embeddings-ingestion
branch: 006-book-embeddings-ingestion
user: Claude Code User
command: Manual ADR Creation
labels: ["architecture", "embeddings", "qdrant", "cohere", "decision"]
links:
  spec: null
  ticket: null
  adr: ADR-001
  pr: null
files_yaml: |
  - history/adr/001-embeddings-and-vector-storage-architecture.md
tests_yaml: |
  - "ADR follows standard template"
  - "Decision clearly documented with rationale"
  - "Alternatives considered and compared"
  - "Consequences identified"
---

# Embeddings Architecture Decision Record

## Prompt Text
```
Architectural decision detected: Cohere embeddings, Qdrant storage, and ingestion pipeline architecture — Document reasoning and tradeoffs?
```

## Response Text
```
Successfully created ADR-001: Embeddings and Vector Storage Architecture in history/adr/001-embeddings-and-vector-storage-architecture.md.

The ADR documents the key architectural decisions for the book content embeddings ingestion pipeline:

1. Embedding Model: Cohere embed-english-v3.0
   - Rationale: Optimized for English technical documentation, good cost-to-quality ratio
   - Alternatives: OpenAI embeddings, Sentence Transformers, multilingual Cohere model

2. Vector Database: Qdrant Cloud
   - Rationale: Cloud-based, good performance, rich filtering, free tier available
   - Alternatives: Pinecone, Weaviate, self-hosted PostgreSQL with pgvector

3. Pipeline Architecture: Single Python script with modular functions
   - Rationale: Simple deployment, cloud-compatible, modular design
   - Alternatives: Microservices, multiple files, serverless functions

The ADR includes context, decision, rationale, alternatives considered, and consequences of each decision. It provides clear documentation of the tradeoffs made in the architecture.
```

## Outcome
The architectural decision record for embeddings and vector storage has been successfully created, documenting the key decisions around Cohere embeddings, Qdrant storage, and the ingestion pipeline architecture with clear rationale and tradeoffs.