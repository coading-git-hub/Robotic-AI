# Contract: Book Content Embeddings Ingestion Interface

## Overview
This contract defines the expected behavior and interfaces for the book content embeddings ingestion pipeline. Since this is a standalone script rather than a web service, the contract focuses on the expected inputs, outputs, and behaviors.

## Functional Specifications

### 1. URL Fetching Interface
**Function**: `get_all_urls()`
- **Input**: None (reads from sitemap.xml at configured BOOK_BASE_URL)
- **Output**: List of URLs (strings)
- **Behavior**: Fetches and parses sitemap.xml to extract all book page URLs
- **Error Handling**: Returns empty list if sitemap is inaccessible

### 2. Content Extraction Interface
**Function**: `extract_text_from_url(url: str)`
- **Input**: Single URL string
- **Output**: Clean text content (string)
- **Behavior**: Fetches HTML from URL and extracts clean text content
- **Error Handling**: Returns empty string if URL is inaccessible

### 3. Text Chunking Interface
**Function**: `chunk_text(content: str)`
- **Input**: Text content string
- **Output**: List of text chunks (strings)
- **Behavior**: Splits content into 512-token chunks with 128-token overlap
- **Error Handling**: Returns empty list if content is invalid

### 4. Embedding Generation Interface
**Function**: `generate_embeddings(chunks: list)`
- **Input**: List of text chunks
- **Output**: List of embedding vectors
- **Behavior**: Generates Cohere embeddings for each chunk
- **Error Handling**: Returns empty list if API calls fail

### 5. Qdrant Storage Interface
**Function**: `store_in_qdrant(embedded_chunks: list, source_url: str, title: str)`
- **Input**: List of embedded chunks, source URL, and title
- **Output**: Boolean success indicator
- **Behavior**: Stores vectors with metadata in Qdrant collection
- **Error Handling**: Returns false if storage fails

## Configuration Interface

### Environment Variables Contract
The script must accept the following environment variables:

- `QDRANT_HOST`: Host URL for Qdrant Cloud instance
- `QDRANT_API_KEY`: API key for Qdrant authentication
- `QDRANT_COLLECTION_NAME`: Name of the collection to store embeddings
- `COHERE_API_KEY`: API key for Cohere embedding service
- `BOOK_BASE_URL`: Base URL for the book site containing sitemap.xml

## Behavioral Contracts

### 1. Idempotency Contract
**Requirement**: Running the script multiple times should not create duplicate vectors
**Implementation**: Use URL + chunk_index as unique identifier in Qdrant

### 2. Error Resilience Contract
**Requirement**: Script continues processing when individual URLs fail
**Implementation**: Log errors but continue with remaining URLs

### 3. Metadata Completeness Contract
**Requirement**: Each stored vector includes complete metadata
**Implementation**: Store URL, title, content preview, and chunk index with each vector

### 4. Progress Tracking Contract
**Requirement**: Script provides feedback during execution
**Implementation**: Log progress as URLs are processed

## Validation Interface

### Validation Functions Contract
The script must provide validation functions:

- `validate_url_fetching(urls: list)`: Verifies URLs are accessible
- `validate_chunk_to_vector_mapping(content: str)`: Verifies chunks match generated vectors
- `validate_metadata_completeness(url: str)`: Verifies metadata is stored correctly