# Tasks: Book Content Embeddings Ingestion Pipeline

**Feature**: Book Content Embeddings Ingestion Pipeline
**Branch**: 006-book-embeddings-ingestion
**Created**: 2025-12-14
**Input**: spec.md, plan.md, research.md, data-model.md, quickstart.md, contracts/ingestion-contract.md

## Implementation Strategy

Implement the book content embeddings ingestion pipeline in priority order: User Story 1 (foundational capability) → User Story 2 (core functionality) → User Story 3 (operational reliability). Each user story is independently testable and delivers value.  

**MVP Scope**: Complete User Story 1 for basic URL fetching, content extraction, and Qdrant storage.

## Dependencies

User stories are designed to be independent but share foundational components. Complete Phase 2 Foundational tasks before starting user stories.

**Story Completion Order**:
1. User Story 1 (P1) - Book Content Ingestion
2. User Story 2 (P2) - Embedding Generation and Storage
3. User Story 3 (P3) - Pipeline Configuration and Repeatability

## Parallel Execution Examples

**Per Story**: Text processing, embedding generation, and storage can be parallelized across different URLs/chunks
**Across Stories**: Environment setup and configuration can happen in parallel with development

---

## Phase 1: Setup

Setup foundational project structure and dependencies for the ingestion pipeline.

- [X] T001 Create main.py in backend directory with basic script structure
- [X] T002 Create requirements.txt with dependencies (requests, beautifulsoup4, cohere, qdrant-client, python-dotenv, langchain)
- [X] T003 Create .env.example with required environment variables
- [X] T004 Set up virtual environment documentation in README

## Phase 2: Foundational Components

Implement foundational components that all user stories depend on.

- [X] T005 [P] Create configuration loading from environment variables
- [X] T006 [P] Implement Qdrant client initialization with error handling
- [X] T007 [P] Implement Cohere client initialization with error handling
- [X] T008 [P] Create utility functions for URL validation and sanitization
- [X] T009 [P] Implement logging setup for the ingestion pipeline

## Phase 3: User Story 1 - Book Content Ingestion (Priority: P1)

Engineers need to run an ingestion pipeline that fetches content from deployed book URLs, processes the HTML into clean text, chunks it into semantic units, generates embeddings, and stores them in Qdrant with proper metadata.

**Independent Test**: Can be fully tested by running the ingestion pipeline on a subset of book URLs and verifying that vectors appear in Qdrant with correct metadata.

- [X] T010 [P] [US1] Implement sitemap.xml fetching function in main.py
- [X] T011 [P] [US1] Create URL extraction from sitemap.xml parser
- [X] T012 [P] [US1] Implement HTML content fetching from individual URLs
- [X] T013 [P] [US1] Create HTML cleaning and text extraction using BeautifulSoup
- [X] T014 [P] [US1] Remove navigation, headers, and non-content elements from HTML
- [X] T015 [P] [US1] Test URL fetching and text extraction with sample URLs
- [X] T016 [US1] Validate that all configured book URLs are fetched successfully

## Phase 4: User Story 2 - Embedding Generation and Storage (Priority: P2)

Engineers need to generate Cohere embeddings for text chunks and store them in Qdrant with complete metadata for downstream retrieval.

**Independent Test**: Can be tested by generating embeddings for a known text sample and verifying they're stored in Qdrant with correct metadata.

- [X] T017 [P] [US2] Implement text chunking function with 512-token size and 128-token overlap
- [X] T018 [P] [US2] Create embedding generation function using Cohere API
- [X] T019 [P] [US2] Implement Qdrant vector storage with metadata (URL, title, content, chunk index)
- [X] T020 [P] [US2] Add rate limiting and retry logic for Cohere API calls
- [X] T021 [P] [US2] Implement error handling for embedding generation failures
- [X] T022 [P] [US2] Test embedding generation and storage with sample content
- [X] T023 [US2] Validate that stored vectors contain correct metadata

## Phase 5: User Story 3 - Pipeline Configuration and Repeatability (Priority: P3)

Engineers need to configure the ingestion pipeline via environment variables and ensure it can be run repeatedly without creating duplicates.

**Independent Test**: Can be tested by running the pipeline multiple times and verifying idempotent behavior (no duplicates).

- [X] T024 [P] [US3] Implement idempotency check using URL + chunk_index as unique identifier
- [X] T025 [P] [US3] Add duplicate prevention in Qdrant storage logic
- [X] T026 [P] [US3] Create progress tracking and resume functionality
- [X] T027 [P] [US3] Add comprehensive error reporting and logging
- [X] T028 [P] [US3] Implement graceful handling of network errors
- [X] T029 [P] [US3] Add command-line arguments for pipeline configuration
- [X] T030 [US3] Test pipeline repeatability without creating duplicate vectors

## Phase 6: Validation & Verification

Implement validation functions to ensure the pipeline meets all requirements.

- [X] T031 [P] Create URL fetching validation function
- [X] T032 [P] Create chunk-to-vector mapping validation function
- [X] T033 [P] Create metadata completeness validation function
- [X] T034 [P] Implement end-to-end pipeline validation test
- [X] T035 [P] Add validation for total vector count matching expected chunks

## Phase 7: Polish & Cross-Cutting Concerns

Final improvements and cross-cutting concerns.

- [X] T036 Add comprehensive error handling throughout the pipeline
- [X] T037 Add performance monitoring and timing metrics
- [X] T038 Create test_embeddings.py for validation and testing
- [X] T039 Add documentation comments to all functions
- [X] T040 Implement memory-efficient processing for large documents
- [X] T041 Add configuration validation at startup
- [X] T042 Create main() function to orchestrate the complete pipeline