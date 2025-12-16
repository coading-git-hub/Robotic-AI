# Feature Specification: Book Content Embeddings Ingestion Pipeline

**Feature Branch**: `006-book-embeddings-ingestion`
**Created**: 2025-12-14
**Status**: Draft
**Input**: User description: "Deploy book URLs, generate embeddings, and persist them in a vector database

Target audience:
- Engineers implementing the RAG data ingestion layer
- Reviewers validating RAG readiness of the book platform

Focus:
- Fetching deployed Docusaurus book content via public URLs
- Cleaning and chunking HTML content into semantically meaningful text units
- Generating embeddings using Cohere embedding models
- Storing vectors and metadata in Qdrant Cloud for downstream retrieval

Success criteria:
- All configured book URLs are fetched successfully
- HTML content is converted to clean, readable text
- Text is chunked with deterministic size and overlap
- Cohere embeddings are generated for every chunk
- Each chunk is stored in Qdrant with:
  - vector embedding
  - source URL
  - page/section title
  - chunk index
- Total vectors in Qdrant equal total generated chunks
- Pipeline is repeatable without duplicating data

Constraints:
- Embedding provider: Cohere
- Vector database: Qdrant Cloud (Free Tier)
- Content source: GitHub Pages–deployed Docusaurus site
- Language: Python
- Configuration via environment variables
- Network-accessible and cloud-compatible execution

Not building:
- Retrieval, similarity search, or reranking
- Question answering or LLM response generation
- Agent logic or OpenAI Agent SDK usage
- Frontend chatbot UI
- Database usage other than Qdrant

Acceptance tests:
- Given a book URL, embeddings are generated and visible in Qdrant
- Stored vectors contain correct metadata
- Re-running ingestion does not create duplicate vectors
- Sample vector can be queried directly from Qdrant for verification
"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Book Content Ingestion (Priority: P1)

Engineers need to run an ingestion pipeline that fetches content from deployed book URLs, processes the HTML into clean text, chunks it into semantic units, generates embeddings, and stores them in Qdrant with proper metadata.

**Why this priority**: This is the foundational capability that enables all downstream RAG functionality - without ingested embeddings, the entire system cannot function.

**Independent Test**: Can be fully tested by running the ingestion pipeline on a subset of book URLs and verifying that vectors appear in Qdrant with correct metadata.

**Acceptance Scenarios**:

1. **Given** deployed Docusaurus book site with sitemap.xml, **When** ingestion pipeline runs, **Then** all book content is fetched, processed, embedded, and stored in Qdrant
2. **Given** book content with HTML formatting, **When** content is processed, **Then** clean text is extracted without HTML tags and metadata is preserved

---

### User Story 2 - Embedding Generation and Storage (Priority: P2)

Engineers need to generate Cohere embeddings for text chunks and store them in Qdrant with complete metadata for downstream retrieval.

**Why this priority**: This enables the core semantic search capability that makes RAG possible - without proper embeddings and metadata, retrieval cannot occur.

**Independent Test**: Can be tested by generating embeddings for a known text sample and verifying they're stored in Qdrant with correct metadata.

**Acceptance Scenarios**:

1. **Given** text chunks from book content, **When** embeddings are generated via Cohere API, **Then** vectors are stored in Qdrant with source URL, title, and chunk index
2. **Given** existing embeddings in Qdrant, **When** ingestion runs again, **Then** no duplicate vectors are created

---

### User Story 3 - Pipeline Configuration and Repeatability (Priority: P3)

Engineers need to configure the ingestion pipeline via environment variables and ensure it can be run repeatedly without creating duplicates.

**Why this priority**: This ensures operational reliability and cloud compatibility - the system must work consistently in different environments.

**Independent Test**: Can be tested by running the pipeline multiple times and verifying idempotent behavior (no duplicates).

**Acceptance Scenarios**:

1. **Given** environment variables for Qdrant and Cohere, **When** pipeline runs, **Then** it connects to services without hardcoding credentials
2. **Given** existing embeddings in Qdrant, **When** pipeline runs again, **Then** no duplicate vectors are created

---

### Edge Cases

- What happens when a book URL returns a 404 or is temporarily unavailable?
- How does the system handle extremely large pages that exceed API limits?
- What if the Cohere API rate limits are exceeded during processing?
- How does the system handle malformed HTML or encoding issues in book content?
- What happens if Qdrant is temporarily unavailable during ingestion?

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST fetch all book URLs from the deployed sitemap.xml
- **FR-002**: System MUST extract clean text from HTML content, removing navigation, headers, and other non-content elements
- **FR-003**: System MUST chunk text into semantic units of 512 tokens with 128-token overlap
- **FR-004**: System MUST generate Cohere embeddings for each text chunk
- **FR-005**: System MUST store each embedding in Qdrant with complete metadata (URL, title, content, chunk index)
- **FR-006**: System MUST be idempotent - re-running the pipeline MUST NOT create duplicate vectors
- **FR-007**: System MUST be configurable via environment variables for Qdrant and Cohere credentials
- **FR-008**: System MUST handle network errors gracefully and continue processing remaining URLs
- **FR-009**: System MUST validate that all embeddings are properly stored in Qdrant after processing

### Key Entities *(include if feature involves data)*

- **Text Chunk**: Semantic unit of book content with content text, token count, and position in source document
- **Embedding Vector**: Numerical representation of text chunk generated by Cohere embedding model
- **Qdrant Record**: Storage unit containing embedding vector, source URL, page title, and chunk metadata
- **Book URL**: Public URL of a book page from the deployed Docusaurus site

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: All configured book URLs are successfully fetched and processed with >95% success rate
- **SC-002**: Text extraction removes >90% of HTML formatting while preserving semantic content
- **SC-003**: Embeddings are generated and stored in Qdrant with 100% of expected metadata fields
- **SC-004**: Pipeline completes without creating duplicate vectors when run multiple times
- **SC-005**: System handles network failures gracefully with appropriate error reporting
- **SC-006**: All configuration is managed through environment variables with no hardcoded values
