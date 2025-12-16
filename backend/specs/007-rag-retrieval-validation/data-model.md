# Data Model: RAG Retrieval Pipeline Validation

**Feature**: 007-rag-retrieval-validation
**Created**: 2025-12-13
**Status**: Draft

## Entity: QueryVector

### Description
Represents the vector embedding of a user query, generated using the same Cohere model as the content ingestion pipeline.

### Fields
- `vector_id` (string): Unique identifier for the query vector
- `vector` (list[float]): The numerical embedding vector from Cohere (1024 dimensions)
- `original_query` (string): The original user query text
- `query_type` (string): Classification of query type (factual, conceptual, etc.)
- `created_at` (datetime): Timestamp when the query was embedded
- `model_version` (string): Version of the embedding model used

### Validation Rules
- `vector` must be a list of floats with exactly 1024 elements
- `original_query` must not be empty
- `vector_id` must be unique

### Relationships
- One QueryVector maps to many RetrievedChunk (one-to-many)

## Entity: RetrievedChunk

### Description
Represents a content chunk retrieved from Qdrant that matches the query vector, with metadata and similarity information.

### Fields
- `chunk_id` (string): Unique identifier for the chunk (content hash from storage)
- `content` (string): The actual text content of the chunk
- `url` (string): Source URL where the content was originally extracted from
- `title` (string): Page/section title from the source content
- `chunk_index` (int): Sequential index of the chunk within the source document
- `similarity_score` (float): Cosine similarity score between query and chunk (0.0-1.0)
- `source_document` (string): Identifier for the original document
- `retrieved_at` (datetime): Timestamp when the chunk was retrieved

### Validation Rules
- `similarity_score` must be between 0.0 and 1.0
- `url`, `title`, and `chunk_index` must be present (not null)
- `content` must not be empty
- `similarity_score` should be above the configured threshold

### Relationships
- One RetrievedChunk belongs to one QueryVector (many-to-one)
- One RetrievedChunk is derived from one stored VectorRecord (many-to-one)

## Entity: ValidationResult

### Description
Records the validation results for a retrieval operation, including accuracy metrics and validation status.

### Fields
- `validation_id` (string): Unique identifier for the validation record
- `query_vector_ref` (string): Reference to the QueryVector being validated
- `retrieved_chunks` (list[dict]): List of retrieved chunks with validation details
- `total_retrieved` (int): Total number of chunks retrieved
- `relevant_count` (int): Number of chunks validated as relevant
- `accuracy_score` (float): Overall accuracy of retrieval (0.0-1.0)
- `metadata_complete` (bool): Whether all chunks have complete metadata
- `validation_passed` (bool): Overall validation status
- `validation_timestamp` (datetime): When validation was performed
- `relevance_threshold` (float): Threshold used for relevance determination

### Validation Rules
- `accuracy_score` must be between 0.0 and 1.0
- `total_retrieved` must be non-negative
- `relevant_count` must be <= `total_retrieved`
- `validation_passed` should be consistent with other validation metrics

### Relationships
- One ValidationResult belongs to one QueryVector (one-to-one)

## Entity: RetrievalSession

### Description
Represents a complete retrieval session with performance metrics and error handling information.

### Fields
- `session_id` (string): Unique identifier for the retrieval session
- `query` (string): Original user query
- `query_vector` (list[float]): Embedded query vector
- `retrieved_chunks` (list[RetrievedChunk]): List of retrieved chunks
- `execution_time_ms` (int): Time taken for retrieval in milliseconds
- `status` (string): Status of retrieval (success, error, timeout)
- `error_message` (string): Error details if status is error
- `top_k_requested` (int): Number of top results requested
- `similarity_threshold` (float): Threshold used for filtering
- `session_timestamp` (datetime): When the session was created

### Validation Rules
- `execution_time_ms` must be non-negative
- `status` must be one of: success, error, timeout
- `top_k_requested` must be positive
- `similarity_threshold` must be between 0.0 and 1.0

### Relationships
- One RetrievalSession contains many RetrievedChunk (one-to-many)

## Collection Schema: Qdrant Search Parameters

### Description
The parameters used for querying the Qdrant vector database during retrieval.

### Search Configuration
- `collection_name`: "book_embeddings" (same as ingestion)
- `query_vector`: 1024-dimensional vector from Cohere
- `limit`: top-k results (default: 5)
- `score_threshold`: minimum similarity (default: 0.3)
- `with_payload`: True (to include metadata)
- `with_vectors`: False (vectors not needed in response)

### Payload Filters
- `url`: Source URL of the content
- `title`: Page/section title
- `chunk_index`: Position in original document
- `source_document`: Identifier for original document
- `content_hash`: Hash of original content for deduplication