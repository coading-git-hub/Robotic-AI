"""
Book Content Embeddings Ingestion Pipeline

This script fetches content from deployed book URLs, processes it (cleans and chunks),
generates Cohere embeddings, and stores them in Qdrant with metadata.

Configuration is done via environment variables:
- COHERE_API_KEY: Cohere API key for embeddings
- QDRANT_URL: Qdrant Cloud cluster URL
- QDRANT_API_KEY: Qdrant API key
- BOOK_BASE_URL: Base URL of deployed book site (default: https://robotic-ai-zlv7.vercel.app)
- CHUNK_SIZE: Size of text chunks (default: 512)
- CHUNK_OVERLAP: Overlap between chunks (default: 128)
"""

import os
import requests
import time
import hashlib
import logging
import uuid
from typing import List, Dict, Any, Optional
from urllib.parse import urljoin, urlparse
from xml.etree import ElementTree as ET

from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
import cohere
from qdrant_client import QdrantClient
from qdrant_client.http import models
from dotenv import load_dotenv


# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_all_urls() -> List[str]:
    """
    Fetch and return all book content URLs from the deployed site.

    Uses the sitemap.xml to discover all content URLs.

    Returns:
        List of valid content URLs
    """
    sitemap_url = os.getenv('BOOK_BASE_URL', 'https://robotic-ai-zlv7.vercel.app') + '/sitemap.xml'
    logger.info(f"Fetching sitemap from {sitemap_url}")

    try:
        response = requests.get(sitemap_url, timeout=30)
        response.raise_for_status()

        # Parse the sitemap XML
        root = ET.fromstring(response.content)

        # Extract URLs from the sitemap
        urls = []
        for url_element in root.findall('.//{http://www.sitemaps.org/schemas/sitemap/0.9}url'):
            loc_element = url_element.find('{http://www.sitemaps.org/schemas/sitemap/0.9}loc')
            if loc_element is not None:
                url = loc_element.text.strip()
                # Filter to include only book content pages
                if '/docs/' in url or '/blog/' in url:  # Adjust based on actual site structure
                    urls.append(url)

        logger.info(f"Found {len(urls)} URLs in sitemap")
        return urls

    except requests.RequestException as e:
        logger.error(f"Failed to fetch sitemap: {e}")
        raise
    except ET.ParseError as e:
        logger.error(f"Failed to parse sitemap XML: {e}")
        raise


def extract_text_from_url(url: str) -> str:
    """
    Extract clean, readable text from a given URL.

    Args:
        url: The URL to extract text from

    Returns:
        Clean text content
    """
    logger.info(f"Extracting text from {url}")

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
            element.decompose()

        # Try to find main content area (adjust selectors based on actual site structure)
        main_content = soup.find('main') or soup.find('article') or soup.find('div', class_='container') or soup.find('div', class_='docItemContainer') or soup.find('div', class_='theme-doc-markdown')

        if main_content:
            text = main_content.get_text(separator=' ', strip=True)
        else:
            # Fallback to body content if main content not found
            body = soup.find('body')
            if body:
                text = body.get_text(separator=' ', strip=True)
            else:
                text = soup.get_text(separator=' ', strip=True)

        # Clean up the text
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)

        logger.info(f"Extracted {len(text)} characters from {url}")
        return text

    except requests.RequestException as e:
        logger.error(f"Failed to fetch URL {url}: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to extract text from {url}: {e}")
        raise


def chunk_text(content: str) -> List[Dict[str, Any]]:
    """
    Split content into semantically meaningful chunks.

    Args:
        content: The content to chunk

    Returns:
        List of chunk dictionaries with content, metadata, and hash
    """
    chunk_size = int(os.getenv('CHUNK_SIZE', '512'))
    chunk_overlap = int(os.getenv('CHUNK_OVERLAP', '128'))

    logger.info(f"Chunking content of {len(content)} characters (size={chunk_size}, overlap={chunk_overlap})")

    # Initialize the text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )

    # Split the content
    chunks = text_splitter.split_text(content)

    # Create chunk dictionaries with metadata
    chunk_dicts = []
    for i, chunk in enumerate(chunks):
        chunk_dict = {
            'content': chunk,
            'chunk_index': i,
            'hash': hashlib.sha256(chunk.encode()).hexdigest()
        }
        chunk_dicts.append(chunk_dict)

    logger.info(f"Created {len(chunk_dicts)} chunks from content")
    return chunk_dicts


def generate_embeddings(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Generate Cohere embeddings for each text chunk.

    Args:
        chunks: List of chunk dictionaries

    Returns:
        Same list with embedding vectors added
    """
    cohere_api_key = os.getenv('COHERE_API_KEY')
    if not cohere_api_key:
        raise ValueError("COHERE_API_KEY environment variable not set")

    logger.info(f"Generating embeddings for {len(chunks)} chunks")

    # Initialize Cohere client
    co = cohere.Client(cohere_api_key)

    # Extract just the content for embedding
    texts = [chunk['content'] for chunk in chunks]

    # Generate embeddings in batches to respect API limits
    batch_size = 96  # Cohere allows up to 96 texts per request
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        # Handle rate limiting with exponential backoff
        max_retries = 5
        for attempt in range(max_retries):
            try:
                response = co.embed(
                    texts=batch,
                    model='embed-english-v3.0',  # Using the recommended model
                    input_type='search_document'  # Appropriate for document search
                )

                all_embeddings.extend(response.embeddings)
                logger.info(f"Processed batch {i//batch_size + 1}/{(len(texts) - 1)//batch_size + 1}")
                break  # Success, exit retry loop

            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) + (1 if attempt > 0 else 0)  # Exponential backoff
                    logger.warning(f"Embedding API error (attempt {attempt + 1}): {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Failed to generate embeddings after {max_retries} attempts: {e}")
                    raise

    # Add embeddings back to the chunk dictionaries
    for i, chunk in enumerate(chunks):
        chunk['vector'] = all_embeddings[i]

    logger.info(f"Successfully generated embeddings for {len(chunks)} chunks")
    return chunks


def store_in_qdrant(embedded_chunks: List[Dict[str, Any]], url: str, title: str) -> bool:
    """
    Store embedded chunks in Qdrant vector database.

    Args:
        embedded_chunks: List of embedded chunks with metadata
        url: Source URL for metadata
        title: Page title for metadata

    Returns:
        Success status (True/False)
    """
    qdrant_url = os.getenv('QDRANT_URL')
    qdrant_api_key = os.getenv('QDRANT_API_KEY')

    if not qdrant_url or not qdrant_api_key:
        raise ValueError("QDRANT_URL and QDRANT_API_KEY environment variables must be set")

    logger.info(f"Storing {len(embedded_chunks)} vectors in Qdrant")

    try:
        # Initialize Qdrant client
        client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
            timeout=60
        )

        # Define collection name
        collection_name = "book_embeddings"

        # Check if collection exists, create if it doesn't
        collection_exists = False
        try:
            client.get_collection(collection_name)
            collection_exists = True
            logger.info(f"Collection {collection_name} already exists")
        except:
            logger.info(f"Creating new collection: {collection_name}")
            client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=1024,  # Cohere embed-english-v3.0 returns 1024-dim vectors
                    distance=models.Distance.COSINE
                )
            )

        # Prepare points for insertion
        points = []
        for chunk in embedded_chunks:
            # Convert hash to a proper UUID format for Qdrant
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk['hash']))
            points.append(
                models.PointStruct(
                    id=point_id,  # Use UUID derived from content hash as the ID for deduplication
                    vector=chunk['vector'],
                    payload={
                        'url': url,
                        'title': title,
                        'chunk_index': chunk['chunk_index'],
                        'content': chunk['content'],
                        'content_hash': chunk['hash'],
                        'source_document': urlparse(url).path
                    }
                )
            )

        # Upsert the points (this will update if exists, insert if new)
        if points:
            client.upsert(
                collection_name=collection_name,
                points=points
            )

        logger.info(f"Successfully stored {len(points)} vectors in Qdrant collection: {collection_name}")
        return True

    except Exception as e:
        logger.error(f"Failed to store in Qdrant: {e}")
        return False


def validate_url_fetching(urls: List[str]) -> bool:
    """
    Validate that all URLs can be successfully fetched.

    Args:
        urls: List of URLs to validate

    Returns:
        Success status
    """
    success_count = 0
    for url in urls:
        try:
            response = requests.head(url, timeout=10)  # Use HEAD to check availability
            if response.status_code == 200:
                success_count += 1
            else:
                logger.warning(f"URL returned status {response.status_code}: {url}")
        except Exception as e:
            logger.warning(f"Failed to validate URL {url}: {e}")

    success_rate = success_count / len(urls) if urls else 0
    logger.info(f"URL validation: {success_count}/{len(urls)} URLs accessible ({success_rate:.2%})")
    return success_rate > 0.9  # Require 90% success rate


def validate_chunk_to_vector_mapping(chunks_count: int, url: str) -> bool:
    """
    Validate that the number of chunks matches the number of stored vectors.

    Args:
        chunks_count: Number of chunks generated
        url: Source URL to verify against

    Returns:
        Success status
    """
    qdrant_url = os.getenv('QDRANT_URL')
    qdrant_api_key = os.getenv('QDRANT_API_KEY')

    if not qdrant_url or not qdrant_api_key:
        logger.error("QDRANT_URL and QDRANT_API_KEY not set for validation")
        return False

    try:
        client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
            timeout=60
        )

        # Count vectors with the specific URL in the collection
        collection_name = "book_embeddings"
        count_result = client.count(
            collection_name=collection_name,
            count_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="url",
                        match=models.MatchValue(value=url)
                    )
                ]
            )
        )

        stored_count = count_result.count
        logger.info(f"Stored vectors for {url}: {stored_count}, Generated chunks: {chunks_count}")

        # They should match (or be very close, accounting for possible deduplication)
        return abs(stored_count - chunks_count) <= 2  # Allow small variance

    except Exception as e:
        logger.error(f"Failed to validate chunk-to-vector mapping: {e}")
        return False


def validate_metadata_completeness(url: str) -> bool:
    """
    Validate that stored vectors have complete metadata.

    Args:
        url: Source URL to check metadata for

    Returns:
        Success status
    """
    qdrant_url = os.getenv('QDRANT_URL')
    qdrant_api_key = os.getenv('QDRANT_API_KEY')

    if not qdrant_url or not qdrant_api_key:
        logger.error("QDRANT_URL and QDRANT_API_KEY not set for validation")
        return False

    try:
        client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
            timeout=60
        )

        # Get a sample of vectors with the specific URL
        collection_name = "book_embeddings"
        records, _ = client.scroll(
            collection_name=collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="url",
                        match=models.MatchValue(value=url)
                    )
                ]
            ),
            limit=5  # Just check a few samples
        )

        if not records:
            logger.warning(f"No records found for URL: {url}")
            return False

        # Check if all required metadata fields are present
        required_fields = ['url', 'title', 'chunk_index', 'content_hash']
        for record in records:
            payload = record.payload
            for field in required_fields:
                if field not in payload:
                    logger.error(f"Missing metadata field '{field}' in record: {record.id}")
                    return False

        logger.info(f"All metadata fields present for {len(records)} sample records from {url}")
        return True

    except Exception as e:
        logger.error(f"Failed to validate metadata completeness: {e}")
        return False


def main():
    """
    Main function to orchestrate the complete ingestion pipeline.
    """
    logger.info("Starting book content embeddings ingestion pipeline")

    # Load configuration
    base_url = os.getenv('BOOK_BASE_URL', 'https://robotic-ai-zlv7.vercel.app')
    logger.info(f"Using base URL: {base_url}")

    try:
        # Step 1: Get all URLs
        urls = get_all_urls()
        if not urls:
            logger.error("No URLs found, exiting")
            return

        logger.info(f"Processing {len(urls)} URLs")

        # Process each URL
        total_chunks = 0
        processed_urls = 0

        for i, url in enumerate(urls):
            logger.info(f"Processing URL {i+1}/{len(urls)}: {url}")

            try:
                # Step 2: Extract text from URL
                content = extract_text_from_url(url)

                if not content.strip():
                    logger.warning(f"No content extracted from {url}, skipping")
                    continue

                # Extract title from the content or URL
                title = urlparse(url).path.split('/')[-1].replace('-', ' ').replace('_', ' ').title()

                # Step 3: Chunk the text
                chunks = chunk_text(content)

                # Step 4: Generate embeddings
                embedded_chunks = generate_embeddings(chunks)

                # Step 5: Store in Qdrant
                success = store_in_qdrant(embedded_chunks, url, title)

                if success:
                    # Step 6: Validate this URL's processing
                    validate_chunk_to_vector_mapping(len(chunks), url)
                    validate_metadata_completeness(url)

                    total_chunks += len(chunks)
                    processed_urls += 1
                    logger.info(f"Successfully processed {url}")
                else:
                    logger.error(f"Failed to store embeddings for {url}")

            except Exception as e:
                logger.error(f"Error processing {url}: {e}")
                continue  # Continue with next URL

        logger.info(f"Pipeline completed. Processed {processed_urls}/{len(urls)} URLs, {total_chunks} total chunks")

        # Final validation
        logger.info("Running final validation checks...")
        if urls:
            validate_url_fetching(urls[:5])  # Validate a sample of URLs

    except Exception as e:
        logger.error(f"Ingestion pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()