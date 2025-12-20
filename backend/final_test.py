#!/usr/bin/env python3
"""
Final test of the actual HTTP server with correct data.
"""
import requests
import time

def final_test():
    print("Final test of the actual HTTP server...")

    # Wait a moment for server to be ready
    time.sleep(2)

    # Test health endpoint first
    print("\n1. Testing health endpoint...")
    try:
        response = requests.get("http://localhost:8002/api/health", timeout=10)
        print(f"   Health endpoint - Status: {response.status_code}")
        print(f"   Response: {response.text}")
    except Exception as e:
        print(f"   Health endpoint error: {e}")

    # Test auth endpoints with shorter password (under 72 bytes)
    print("\n2. Testing auth endpoints...")

    # Test signup with shorter password
    print("   Testing signup endpoint with valid data...")
    signup_data = {
        "email": "test@example.com",
        "password": "shortpass123",  # Shorter password
        "name": "Test User"
    }
    try:
        response = requests.post(
            "http://localhost:8002/api/auth/signup",
            json=signup_data,
            timeout=10
        )
        print(f"   Signup endpoint - Status: {response.status_code}")
        print(f"   Response: {response.text}")
    except Exception as e:
        print(f"   Signup endpoint error: {e}")

    # Test signin (should work if signup succeeded)
    print("   Testing signin endpoint...")
    signin_data = {
        "email": "test@example.com",
        "password": "shortpass123"
    }
    try:
        response = requests.post(
            "http://localhost:8002/api/auth/signin",
            json=signin_data,
            timeout=10
        )
        print(f"   Signin endpoint - Status: {response.status_code}")
        print(f"   Response: {response.text}")
    except Exception as e:
        print(f"   Signin endpoint error: {e}")

    # Test RAG agent endpoint
    print("\n3. Testing RAG agent endpoint...")
    query_data = {
        "query": "test query",
        "selected_text": "test selected text"
    }
    try:
        response = requests.post(
            "http://localhost:8002/api/agent/query",
            json=query_data,
            timeout=15
        )
        print(f"   Agent query endpoint - Status: {response.status_code}")
        print(f"   Response: {response.text}")
    except Exception as e:
        print(f"   Agent query endpoint error: {e}")

    print("\n4. Final test completed!")

if __name__ == "__main__":
    final_test()