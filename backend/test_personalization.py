#!/usr/bin/env python3
"""
Test personalization endpoints to see if they work.
"""
import requests

def test_personalization_endpoints():
    print("Testing personalization endpoints...")

    # Test personalization endpoint
    print("Testing personalization endpoint...")
    try:
        response = requests.post(
            "http://localhost:8002/api/personalization/personalize",
            json={"query": "test", "user_id": "test"},
            timeout=10
        )
        print(f"   Personalize endpoint - Status: {response.status_code}")
        print(f"   Response: {response.text}")
    except Exception as e:
        print(f"   Personalize endpoint error: {e}")

    # Test history endpoint
    print("Testing history endpoint...")
    try:
        response = requests.get(
            "http://localhost:8002/api/personalization/history",
            timeout=10
        )
        print(f"   History endpoint - Status: {response.status_code}")
        print(f"   Response: {response.text}")
    except Exception as e:
        print(f"   History endpoint error: {e}")

if __name__ == "__main__":
    test_personalization_endpoints()