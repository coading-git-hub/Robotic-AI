#!/usr/bin/env python3
"""
Test with a very short password to bypass bcrypt initialization issues.
"""
import requests
import time
import json

def test_very_short_password():
    print("Testing with very short passwords...")

    # Test with a 4-character password (should work now)
    print("\n1. Testing signup with 4-character password...")
    signup_data = {
        "email": f"testuser{int(time.time())}@example.com",
        "password": "1234",  # 4 characters
        "name": "Test User"
    }

    try:
        response = requests.post(
            "http://localhost:8002/api/auth/signup",
            headers={"Content-Type": "application/json"},
            data=json.dumps(signup_data),
            timeout=10
        )
        print(f"   4-char password - Status: {response.status_code}")
        print(f"   Response: {response.text}")
    except Exception as e:
        print(f"   4-char password error: {e}")

    # Test with a 5-character password (should work)
    print("\n2. Testing signup with 5-character password...")
    signup_data_5 = {
        "email": f"testuser2{int(time.time())}@example.com",
        "password": "12345",  # 5 characters
        "name": "Test User 2"
    }

    try:
        response = requests.post(
            "http://localhost:8002/api/auth/signup",
            headers={"Content-Type": "application/json"},
            data=json.dumps(signup_data_5),
            timeout=10
        )
        print(f"   5-char password - Status: {response.status_code}")
        print(f"   Response: {response.text}")
    except Exception as e:
        print(f"   5-char password error: {e}")

    print("\nShort password tests completed!")

if __name__ == "__main__":
    test_very_short_password()