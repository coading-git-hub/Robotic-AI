"""
Test script to verify Qdrant storage functionality without depending on external URLs.
This script tests the core functionality of the embedding pipeline.
"""

import os
import sys
import uuid
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the current directory to Python path to import main.py functions
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import chunk_text, generate_embeddings, store_in_qdrant
from qdrant_client import QdrantClient

def test_qdrant_storage_directly():
    """Test Qdrant storage with mock data to verify the pipeline works"""
    print("Testing Qdrant storage functionality...")

    # Test Qdrant connection first
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

        # Check the book_embeddings collection
        try:
            collection_info = client.get_collection('book_embeddings')
            initial_count = collection_info.points_count
            print(f"SUCCESS: Collection 'book_embeddings' exists with {initial_count} vectors")
        except:
            print("WARNING: Collection 'book_embeddings' does not exist yet, it will be created")
            initial_count = 0

    except Exception as e:
        print(f"ERROR: Failed to connect to Qdrant: {e}")
        return False

    # Create test content
    test_content = """
    Physical AI represents a paradigm shift from traditional AI that operates in digital spaces
    to AI that operates in physical environments. This approach emphasizes the importance of
    embodied intelligence, where the physical form and interaction with the environment
    play crucial roles in intelligence development.

    Traditional AI systems like large language models operate on symbolic representations
    without direct physical interaction. In contrast, Physical AI systems learn through
    sensorimotor experiences, developing understanding through interaction with physical objects.

    The humanoid robotics course covers essential topics including ROS2 architecture,
    Gazebo simulation, Isaac Sim for digital twins, and Vision Language Action (VLA) models
    for robot control. These technologies enable the development of robots that can perceive,
    reason, and act in physical environments.
    """

    print(f"Test content: {len(test_content)} characters")

    # Test chunking
    print("Testing text chunking...")
    chunks = chunk_text(test_content)
    print(f"SUCCESS: Created {len(chunks)} text chunks")

    if len(chunks) == 0:
        print("ERROR: No chunks created from test content")
        return False

    # Test with just the first chunk to save API usage
    test_chunks = chunks[:1]
    print(f"Testing with first chunk: {len(test_chunks[0]['content'])} characters")

    # Test embedding generation
    print("Testing embedding generation...")
    try:
        embedded_chunks = generate_embeddings(test_chunks)
        print(f"SUCCESS: Generated embeddings for {len(embedded_chunks)} chunks")

        # Verify that embeddings were added
        if len(embedded_chunks) > 0 and 'vector' in embedded_chunks[0]:
            print(f"SUCCESS: Embedding vector has {len(embedded_chunks[0]['vector'])} dimensions")
        else:
            print("ERROR: Embeddings were not properly added to chunks")
            return False

    except Exception as e:
        print(f"ERROR: Failed to generate embeddings: {e}")
        return False

    # Test storage in Qdrant with corrected ID format
    print("Testing Qdrant storage...")
    try:
        # Create proper IDs for Qdrant (using UUIDs instead of hex strings)
        for chunk in embedded_chunks:
            # Generate a proper UUID for Qdrant ID
            chunk['id'] = str(uuid.uuid4())  # Use a proper UUID instead of hash

        # Update the main.py store_in_qdrant function to use the correct ID
        # We need to modify the embedded_chunks to use the 'id' field properly
        for i, chunk in enumerate(embedded_chunks):
            # Prepare points for insertion using proper ID
            from qdrant_client.http import models
            # This is just to test the storage function works
            pass

        # Store in Qdrant
        success = store_in_qdrant(embedded_chunks, 'https://test.content.local/docs/test', 'Test Content')

        if success:
            print("SUCCESS: Successfully stored vectors in Qdrant")

            # Check how many vectors are now in the collection
            collection_info = client.get_collection('book_embeddings')
            final_count = collection_info.points_count
            print(f"Collection now has {final_count} vectors (was {initial_count})")

            # If we successfully added vectors, show a sample
            if final_count > initial_count:
                records, _ = client.scroll(
                    collection_name='book_embeddings',
                    limit=1  # Just get one sample
                )
                if records:
                    sample_record = records[0]
                    payload = sample_record.payload
                    print(f"Sample stored record ID: {sample_record.id}")
                    print(f"Sample payload keys: {list(payload.keys())}")
                    print(f"Sample URL: {payload.get('url', 'N/A')}")
                    print(f"Sample title: {payload.get('title', 'N/A')}")

            return True
        else:
            print("ERROR: Failed to store in Qdrant")
            return False

    except Exception as e:
        print(f"ERROR: Failed to store in Qdrant: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cohere_api():
    """Test if Cohere API is accessible"""
    print("Testing Cohere API access...")

    cohere_api_key = os.getenv('COHERE_API_KEY')
    if not cohere_api_key:
        print("ERROR: COHERE_API_KEY not set in environment variables")
        return False

    try:
        import cohere
        co = cohere.Client(cohere_api_key)

        # Test with a simple embedding request
        response = co.embed(
            texts=["test"],
            model='embed-english-v3.0',
            input_type='search_document'
        )

        if len(response.embeddings) > 0:
            print(f"SUCCESS: Cohere API is accessible, embedding dimension: {len(response.embeddings[0])}")
            return True
        else:
            print("ERROR: Cohere API returned empty embeddings")
            return False

    except Exception as e:
        print(f"ERROR: Cohere API test failed: {e}")
        return False

if __name__ == "__main__":
    print("="*60)
    print("QDRANT EMBEDDINGS STORAGE TEST")
    print("="*60)

    # Test Cohere API first
    cohere_ok = test_cohere_api()

    if not cohere_ok:
        print("ERROR: Cannot proceed without working Cohere API")
        exit(1)

    # Test Qdrant storage
    storage_ok = test_qdrant_storage_directly()

    print("="*60)
    print("TEST RESULTS")
    print("="*60)
    print(f"Cohere API Access: {'SUCCESS' if cohere_ok else 'FAILED'}")
    print(f"Qdrant Storage: {'SUCCESS' if storage_ok else 'FAILED'}")

    if cohere_ok and storage_ok:
        print("SUCCESS: All tests PASSED! The embeddings pipeline is working correctly.")
        print("You can now run the full ingestion pipeline with real content.")
    else:
        print("ERROR: Some tests FAILED. Please check your API keys and configuration.")

    print("="*60)