#!/usr/bin/env python3
"""
Test the unified API using FastAPI TestClient instead of HTTP requests.
This will help determine if the issue is with the server runtime or the route registration.
"""
import sys
import os

# Add paths
backend_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, backend_dir)
parent_dir = os.path.dirname(backend_dir)
sys.path.insert(0, parent_dir)

def test_with_testclient():
    print("Testing API using FastAPI TestClient...")

    # Import the app
    from src.main import app

    # Import TestClient
    from fastapi.testclient import TestClient

    # Create test client
    client = TestClient(app)

    print("Testing endpoints with TestClient:")

    # Test health endpoint
    print("\n1. Testing health endpoint...")
    response = client.get("/api/health")
    print(f"   Health endpoint - Status: {response.status_code}")
    print(f"   Response: {response.text}")

    # Test auth endpoints
    print("\n2. Testing auth endpoints...")

    # Test signup
    print("   Testing signup endpoint...")
    signup_data = {
        "email": "test@example.com",
        "password": "testpassword123",
        "name": "Test User"
    }
    response = client.post("/api/auth/signup", json=signup_data)
    print(f"   Signup endpoint - Status: {response.status_code}")
    print(f"   Response: {response.text}")

    # Test signin
    print("   Testing signin endpoint...")
    signin_data = {
        "email": "test@example.com",
        "password": "testpassword123"
    }
    response = client.post("/api/auth/signin", json=signin_data)
    print(f"   Signin endpoint - Status: {response.status_code}")
    print(f"   Response: {response.text}")

    # Test RAG agent endpoint
    print("\n3. Testing RAG agent endpoint...")
    query_data = {
        "query": "test query",
        "selected_text": "test selected text"
    }
    response = client.post("/api/agent/query", json=query_data)
    print(f"   Agent query endpoint - Status: {response.status_code}")
    print(f"   Response: {response.text}")

    print("\n4. All tests completed with TestClient!")

if __name__ == "__main__":
    test_with_testclient()