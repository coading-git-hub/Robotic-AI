# Qdrant Vector Database Setup

This document outlines how to set up Qdrant Cloud for the RAG system.

## Cloud Setup

### 1. Create Qdrant Cloud Account
- Go to https://cloud.qdrant.io/
- Sign up for a free account or log in if you already have one
- Create a new cluster with appropriate specifications for your needs

### 2. Configure Collection for Book Content
Create a collection with the following specifications:

```json
{
  "collection_name": "book_content",
  "vector_size": 1536,
  "distance": "Cosine",
  "hnsw_config": {
    "m": 16,
    "ef_construct": 100,
    "full_scan_threshold": 10000
  },
  "optimizers_config": {
    "deleted_threshold": 0.2,
    "vacuum_min_vector_number": 1000,
    "default_segment_number": 2,
    "max_segment_size": 100000,
    "memmap_threshold": 10000,
    "indexing_threshold": 20000
  }
}
```

### 3. Environment Configuration
Add the following to your `.env` file:

```env
QDRANT_URL=your_cluster_endpoint_url
QDRANT_API_KEY=your_api_key
QDRANT_COLLECTION_NAME=book_content
```

### 4. Local Development Setup (Alternative)
If you prefer to run Qdrant locally for development:

1. Install Docker
2. Run Qdrant with Docker:
```bash
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant
```

3. Use this configuration in your `.env`:
```env
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
```

## Collection Schema

The collection should have the following payload schema:

```json
{
  "content_id": { "type": "keyword" },
  "content_type": { "type": "keyword" },
  "text_content": { "type": "text" },
  "source_module": { "type": "keyword" },
  "source_lesson": { "type": "keyword" },
  "token_count": { "type": "integer" },
  "created_at": { "type": "datetime" },
  "updated_at": { "type": "datetime" }
}
```

## API Usage

The backend will interact with Qdrant through the following operations:
- Upload content chunks with embeddings
- Perform semantic search on queries
- Update and delete content as needed
- Manage collection metadata