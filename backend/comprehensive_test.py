#!/usr/bin/env python3
"""
Comprehensive test of the unified API endpoints.
"""
import time
import requests
import subprocess
import os
import signal
import sys

def test_all_endpoints():
    print("Testing all unified API endpoints...")

    # Test health endpoint first
    print("\n1. Testing health endpoint...")
    try:
        response = requests.get("http://localhost:8002/api/health", timeout=10)
        print(f"   Health endpoint - Status: {response.status_code}")
        print(f"   Response: {response.text}")
    except Exception as e:
        print(f"   Health endpoint error: {e}")

    # Test auth endpoints
    print("\n2. Testing auth endpoints...")

    # Test signup
    print("   Testing signup endpoint...")
    signup_data = {
        "email": "test@example.com",
        "password": "testpassword123",
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

    # Test signin (with the same user we just created or an existing one)
    print("   Testing signin endpoint...")
    signin_data = {
        "email": "test@example.com",
        "password": "testpassword123"
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
            timeout=15  # Longer timeout for RAG processing
        )
        print(f"   Agent query endpoint - Status: {response.status_code}")
        print(f"   Response: {response.text}")
    except Exception as e:
        print(f"   Agent query endpoint error: {e}")

    print("\n4. All endpoint tests completed!")

if __name__ == "__main__":
    print("Starting unified API endpoint tests...")
    test_all_endpoints()