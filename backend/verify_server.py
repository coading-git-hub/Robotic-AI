#!/usr/bin/env python3
"""
Verify that the server is running and accessible.
"""
import requests
import time

def verify_server():
    print("Verifying server is running and accessible...")

    # Wait a moment for any connections to settle
    time.sleep(1)

    # Test health endpoint
    print("\nTesting health endpoint...")
    try:
        response = requests.get("http://localhost:8002/api/health", timeout=10)
        print(f"[SUCCESS] Health endpoint - Status: {response.status_code}")
        print(f"   Response: {response.text}")
    except Exception as e:
        print(f"[ERROR] Health endpoint error: {e}")

    # Test a simple GET endpoint
    print("\nTesting root endpoint...")
    try:
        response = requests.get("http://localhost:8002/", timeout=5)
        print(f"[SUCCESS] Root endpoint - Status: {response.status_code}")
        print(f"   Response: {response.text}")
    except Exception as e:
        print(f"[ERROR] Root endpoint error: {e}")

    # Test RAG endpoint to confirm it works
    print("\nTesting RAG agent endpoint...")
    query_data = {
        "query": "hello",
        "selected_text": ""
    }
    try:
        response = requests.post(
            "http://localhost:8002/api/agent/query",
            json=query_data,
            timeout=15
        )
        print(f"[SUCCESS] RAG agent endpoint - Status: {response.status_code}")
        print(f"   Response preview: {str(response.text)[:100]}...")
    except Exception as e:
        print(f"[ERROR] RAG agent endpoint error: {e}")

    print("\nServer verification completed!")

if __name__ == "__main__":
    verify_server()