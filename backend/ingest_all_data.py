#!/usr/bin/env python3
"""
Script to run the full book embeddings ingestion with known accessible URLs
since the sitemap contains outdated URLs.
"""

import os
import sys
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Add the current directory to Python path to import main.py functions
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import extract_text_from_url, chunk_text, generate_embeddings, store_in_qdrant

# Known accessible URLs from the Physical AI & Humanoid Robotics course
ACCESSIBLE_URLS = [
    "https://robotic-ai-zlv7.vercel.app/docs/intro",
    "https://robotic-ai-zlv7.vercel.app/docs/week-1-2/physical-ai-intro",
    "https://robotic-ai-zlv7.vercel.app/docs/week-1-2/embodied-intelligence",
    "https://robotic-ai-zlv7.vercel.app/docs/week-1-2/sensor-systems",
    "https://robotic-ai-zlv7.vercel.app/docs/week-3-5/ros2-architecture",
    "https://robotic-ai-zlv7.vercel.app/docs/week-3-5/nodes-topics-services",
    "https://robotic-ai-zlv7.vercel.app/docs/week-3-5/ros2-packages",
    "https://robotic-ai-zlv7.vercel.app/docs/week-6-8/gazebo-simulation",
    "https://robotic-ai-zlv7.vercel.app/docs/week-6-8/isaac-sim",
    "https://robotic-ai-zlv7.vercel.app/docs/week-6-8/unity-rendering",
    "https://robotic-ai-zlv7.vercel.app/docs/week-9-11/isaac-ros",
    "https://robotic-ai-zlv7.vercel.app/docs/week-9-11/sim-to-real",
    "https://robotic-ai-zlv7.vercel.app/docs/week-9-11/vslam-navigation",
    "https://robotic-ai-zlv7.vercel.app/docs/week-12-13/vla-integration",
    "https://robotic-ai-zlv7.vercel.app/docs/week-12-13/autonomous-humanoid",
    "https://robotic-ai-zlv7.vercel.app/docs/week-12-13/capstone-project",
    "https://robotic-ai-zlv7.vercel.app/docs/appendices/ros2-cheatsheet",
    "https://robotic-ai-zlv7.vercel.app/docs/appendices/simulation-tips",
    "https://robotic-ai-zlv7.vercel.app/docs/appendices/troubleshooting",
]

def main():
    print("="*60)
    print("INGESTING ALL BOOK DATA INTO QDRANT")
    print("="*60)

    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    print(f"Found {len(ACCESSIBLE_URLS)} accessible URLs to process")
    print("Starting ingestion process...\n")

    total_chunks = 0
    processed_urls = 0
    failed_urls = 0

    for i, url in enumerate(ACCESSIBLE_URLS):
        print(f"[{i+1}/{len(ACCESSIBLE_URLS)}] Processing: {url}")

        try:
            # Step 1: Extract text from URL
            content = extract_text_from_url(url)

            if not content.strip():
                print(f"  ❌ No content extracted from {url}")
                failed_urls += 1
                continue

            print(f"  SUCCESS: Extracted {len(content)} characters")

            # Extract title from the URL
            from urllib.parse import urlparse
            path_parts = urlparse(url).path.strip('/').split('/')
            if len(path_parts) > 0:
                title = ' '.join(path_parts[-1].split('-')).title()
            else:
                title = "Unknown Title"

            # Step 2: Chunk the text
            chunks = chunk_text(content)
            print(f"  SUCCESS: Created {len(chunks)} chunks")

            if len(chunks) == 0:
                print(f"  ERROR: No chunks created from {url}")
                failed_urls += 1
                continue

            # Step 3: Generate embeddings
            embedded_chunks = generate_embeddings(chunks)
            print(f"  SUCCESS: Generated embeddings for {len(embedded_chunks)} chunks")

            # Step 4: Store in Qdrant
            success = store_in_qdrant(embedded_chunks, url, title)

            if success:
                print(f"  SUCCESS: Stored {len(embedded_chunks)} vectors in Qdrant")
                total_chunks += len(chunks)
                processed_urls += 1
            else:
                print(f"  ERROR: Failed to store embeddings in Qdrant for {url}")
                failed_urls += 1

        except Exception as e:
            print(f"  ERROR: Error processing {url}: {str(e)}")
            failed_urls += 1
            continue

        print()  # Empty line for readability

    print("="*60)
    print("INGESTION COMPLETED")
    print("="*60)
    print(f"Processed URLs: {processed_urls}")
    print(f"Failed URLs: {failed_urls}")
    print(f"Total chunks ingested: {total_chunks}")
    print(f"Total URLs attempted: {len(ACCESSIBLE_URLS)}")

    # Final check of Qdrant collection
    from qdrant_client import QdrantClient
    qdrant_url = os.getenv('QDRANT_URL')
    qdrant_api_key = os.getenv('QDRANT_API_KEY')

    if qdrant_url and qdrant_api_key:
        try:
            client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key, timeout=60)
            collection_info = client.get_collection('book_embeddings')
            print(f"Final vector count in Qdrant: {collection_info.points_count}")
        except Exception as e:
            print(f"Could not check final Qdrant count: {e}")

    print("="*60)
    if failed_urls == 0:
        print("SUCCESS: All URLs processed successfully!")
    elif processed_urls > 0:
        print(f"PARTIAL SUCCESS: {processed_urls} URLs processed, {failed_urls} failed")
    else:
        print("FAILURE: No URLs were successfully processed")
    print("="*60)

if __name__ == "__main__":
    main()