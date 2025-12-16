


























































"""
Test script to verify that all data embeddings are properly stored in Qdrant.
This script will run the embedding pipeline and validate the storage.
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the current directory to Python path to import main.py functions
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import get_all_urls, extract_text_from_url, chunk_text, generate_embeddings, store_in_qdrant, validate_url_fetching, validate_chunk_to_vector_mapping, validate_metadata_completeness, main
from qdrant_client import QdrantClient
from qdrant_client.http import models


def test_qdrant_connection():
    """Test connection to Qdrant"""
    print("Testing Qdrant connection...")

    qdrant_url = os.getenv('QDRANT_URL')
    qdrant_api_key = os.getenv('QDRANT_API_KEY')

    if not qdrant_url or not qdrant_api_key:
        print("ERROR: QDRANT_URL and/or QDRANT_API_KEY not set in environment variables")
        return False

    try:
        client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
            timeout=60
        )

        # Test connection by getting collections
        collections = client.get_collections()
        print(f"SUCCESS: Successfully connected to Qdrant")
        print(f"Available collections: {[col.name for col in collections.collections]}")
        return True

    except Exception as e:
        print(f"ERROR: Failed to connect to Qdrant: {e}")
        return False


def test_sitemap_access():
    """Test access to the sitemap"""
    print("\nTesting sitemap access...")

    try:
        urls = get_all_urls()
        print(f"SUCCESS: Successfully fetched {len(urls)} URLs from sitemap")
        if urls:
            print(f"Sample URLs: {urls[:3]}")  # Show first 3 URLs
        return len(urls) > 0
    except Exception as e:
        print(f"ERROR: Failed to fetch URLs from sitemap: {e}")
        return False


def test_single_url_processing():
    """Test processing of a single URL to verify the full pipeline"""
    print("\nTesting single URL processing pipeline...")

    try:
        # Get URLs from sitemap
        urls = get_all_urls()
        if not urls:
            print("ERROR: No URLs found in sitemap")
            return False

        # Test with the first URL
        test_url = urls[0]
        print(f"Testing with URL: {test_url}")

        # Extract text
        content = extract_text_from_url(test_url)
        print(f"SUCCESS: Extracted {len(content)} characters from URL")

        # Chunk text
        chunks = chunk_text(content)
        print(f"SUCCESS: Created {len(chunks)} chunks")

        # Generate embeddings (only first 3 chunks to save API usage)
        test_chunks = chunks[:3] if len(chunks) > 3 else chunks
        embedded_chunks = generate_embeddings(test_chunks)
        print(f"SUCCESS: Generated embeddings for {len(embedded_chunks)} chunks")

        # Extract title from URL
        from urllib.parse import urlparse
        title = urlparse(test_url).path.split('/')[-1].replace('-', ' ').replace('_', ' ').title()

        # Store in Qdrant
        success = store_in_qdrant(embedded_chunks, test_url, title)
        if success:
            print(f"SUCCESS: Successfully stored {len(embedded_chunks)} vectors in Qdrant")
        else:
            print(f"ERROR: Failed to store vectors in Qdrant")

        return success

    except Exception as e:
        print(f"FAIL: Error in single URL processing: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_qdrant_storage():
    """Check what's currently stored in Qdrant"""
    print("\nChecking Qdrant storage...")

    qdrant_url = os.getenv('QDRANT_URL')
    qdrant_api_key = os.getenv('QDRANT_API_KEY')

    if not qdrant_url or not qdrant_api_key:
        print("❌ QDRANT_URL and/or QDRANT_API_KEY not set")
        return False

    try:
        client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
            timeout=60
        )

        # Check if collection exists
        collection_name = "book_embeddings"
        try:
            collection_info = client.get_collection(collection_name)
            print(f"✅ Collection '{collection_name}' exists")
            print(f"Collection vectors count: {collection_info.points_count}")

            # Get a sample of stored vectors
            if collection_info.points_count > 0:
                sample_records, _ = client.scroll(
                    collection_name=collection_name,
                    limit=3
                )

                print(f"Sample records from Qdrant:")
                for i, record in enumerate(sample_records):
                    payload = record.payload
                    print(f"  Record {i+1}:")
                    print(f"    ID: {record.id}")
                    print(f"    URL: {payload.get('url', 'N/A')}")
                    print(f"    Title: {payload.get('title', 'N/A')}")
                    print(f"    Content preview: {payload.get('content', '')[:100]}...")

            return True

        except Exception as e:
            print(f"❌ Collection '{collection_name}' does not exist: {e}")
            return False

    except Exception as e:
        print(f"❌ Error checking Qdrant storage: {e}")
        return False


def run_full_pipeline_test():
    """Run the full pipeline test"""
    print("="*60)
    print("RUNNING EMBEDDINGS PIPELINE VALIDATION TEST")
    print("="*60)

    # Test 1: Qdrant connection
    qdrant_ok = test_qdrant_connection()

    if not qdrant_ok:
        print("\nERROR: Cannot proceed without Qdrant connection")
        return False

    # Test 2: Sitemap access
    sitemap_ok = test_sitemap_access()

    # Test 3: Storage check
    storage_ok = check_qdrant_storage()

    # Test 4: Single URL processing (if we have URLs)
    if sitemap_ok:
        processing_ok = test_single_url_processing()
    else:
        processing_ok = False

    print("\n" + "="*60)
    print("TEST RESULTS SUMMARY")
    print("="*60)
    print(f"Qdrant Connection: {'PASS' if qdrant_ok else 'FAIL'}")
    print(f"Sitemap Access: {'PASS' if sitemap_ok else 'FAIL'}")
    print(f"Storage Check: {'PASS' if storage_ok else 'FAIL'}")
    print(f"Single URL Processing: {'PASS' if processing_ok else 'FAIL'}")

    all_tests_pass = qdrant_ok and sitemap_ok and storage_ok and processing_ok

    if all_tests_pass:
        print("\nSUCCESS: All tests PASSED! Embeddings pipeline is working correctly.")
        print("SUCCESS: Data is being properly stored in Qdrant")
    else:
        print("\nWARNING: Some tests FAILED. Please check the configuration and try again.")

    return all_tests_pass


def run_validation_functions():
    """Run the validation functions to check data integrity"""
    print("\n" + "="*60)
    print("RUNNING VALIDATION CHECKS")
    print("="*60)

    try:
        # Get URLs for validation
        urls = get_all_urls()
        if not urls:
            print("❌ No URLs available for validation")
            return False

        print(f"Validating with {len(urls)} URLs...")

        # Run validation functions
        url_validation = validate_url_fetching(urls[:5])  # Validate first 5 URLs
        print(f"URL fetching validation: {'PASS' if url_validation else 'FAIL'}")

        # Check Qdrant for stored data
        qdrant_url = os.getenv('QDRANT_URL')
        qdrant_api_key = os.getenv('QDRANT_API_KEY')

        if qdrant_url and qdrant_api_key:
            try:
                from qdrant_client import QdrantClient
                client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key, timeout=60)

                # Count total vectors in collection
                try:
                    collection_info = client.get_collection("book_embeddings")
                    total_vectors = collection_info.points_count
                    print(f"Total vectors in Qdrant: {total_vectors}")

                    if total_vectors > 0:
                        print("SUCCESS: Data is present in Qdrant")

                        # Validate metadata completeness on a sample
                        if len(urls) > 0:
                            metadata_validation = validate_metadata_completeness(urls[0])
                            print(f"Metadata completeness validation: {'✅ PASS' if metadata_validation else '❌ FAIL'}")

                    else:
                        print("⚠️ No vectors found in Qdrant collection")

                except Exception as e:
                    print(f"❌ Error accessing Qdrant collection: {e}")

            except Exception as e:
                print(f"❌ Error creating Qdrant client: {e}")

        return True

    except Exception as e:
        print(f"❌ Error running validation functions: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run the full test
    success = run_full_pipeline_test()

    # Run validation functions
    run_validation_functions()

    print("\n" + "="*60)
    print("EMBEDDINGS VALIDATION COMPLETE")
    print("="*60)

    if success:
        print("✅ All checks completed successfully!")
        print("The embedding pipeline is working and data is being stored in Qdrant.")
    else:
        print("❌ Some issues were found during validation.")
        print("Please check your environment variables and internet connection.")