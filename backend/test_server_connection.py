#!/usr/bin/env python3
"""
Test if the unified API server is running and accessible.
"""
import time
import requests

def test_server():
    print("Testing if unified API server is accessible...")

    # Test different endpoints
    endpoints = [
        "http://localhost:8002",
        "http://localhost:8002/health",
        "http://localhost:8002/api/health",
        "http://localhost:8002/api/auth/signup",
        "http://localhost:8002/api/agent/query"
    ]

    for endpoint in endpoints:
        try:
            if "signup" in endpoint or "query" in endpoint:
                # Use GET to test if the endpoint exists (won't actually signup/query)
                response = requests.get(endpoint, timeout=3)
            else:
                response = requests.get(endpoint, timeout=3)
            print(f"[SUCCESS] {endpoint} - Status: {response.status_code}")
        except requests.exceptions.ConnectionError:
            print(f"[ERROR] {endpoint} - Connection refused (server not running)")
        except requests.exceptions.Timeout:
            print(f"[TIMEOUT] {endpoint} - Timeout (may be processing)")
        except Exception as e:
            print(f"[ERROR] {endpoint} - Error: {str(e)[:100]}...")  # Limit error message length

if __name__ == "__main__":
    test_server()