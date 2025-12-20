#!/usr/bin/env python3
"""
Test signup with short passwords using FastAPI TestClient to bypass potential network/initialization issues.
"""
import sys
import os

# Add paths
backend_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, backend_dir)
parent_dir = os.path.dirname(backend_dir)
sys.path.insert(0, parent_dir)

def test_with_testclient():
    print("Testing signup with short passwords using TestClient...")

    # Import the app
    from src.main import app

    # Import TestClient
    from fastapi.testclient import TestClient

    # Create test client
    client = TestClient(app)

    # Test with 4-character password
    print("\n1. Testing signup with 4-character password using TestClient...")
    signup_data = {
        "email": "test4@example.com",
        "password": "1234",  # 4 characters
        "name": "Test User 4"
    }

    try:
        response = client.post("/api/auth/signup", json=signup_data)
        print(f"   4-char password - Status: {response.status_code}")
        print(f"   Response: {response.text}")
    except Exception as e:
        print(f"   4-char password error: {e}")

    # Test with 6-character password
    print("\n2. Testing signup with 6-character password using TestClient...")
    signup_data_6 = {
        "email": "test6@example.com",
        "password": "123456",  # 6 characters
        "name": "Test User 6"
    }

    try:
        response = client.post("/api/auth/signup", json=signup_data_6)
        print(f"   6-char password - Status: {response.status_code}")
        print(f"   Response: {response.text}")
    except Exception as e:
        print(f"   6-char password error: {e}")

    # Test with 9-character password (should fail with our validation)
    print("\n3. Testing signup with 9-character password using TestClient...")
    signup_data_9 = {
        "email": "test9@example.com",
        "password": "123456789",  # 9 characters
        "name": "Test User 9"
    }

    try:
        response = client.post("/api/auth/signup", json=signup_data_9)
        print(f"   9-char password - Status: {response.status_code}")
        print(f"   Response: {response.text}")
    except Exception as e:
        print(f"   9-char password error: {e}")

    print("\nTestClient tests completed!")

if __name__ == "__main__":
    test_with_testclient()