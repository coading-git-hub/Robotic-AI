#!/usr/bin/env python3
"""
Test the updated 8-character password validation.
"""
import requests
import time
import json

def test_short_password():
    print("Testing 8-character password validation...")

    # Test with an 8-character password (should work)
    print("\n1. Testing signup with 8-character password...")
    signup_data = {
        "email": f"testuser{int(time.time())}@example.com",
        "password": "12345678",  # Exactly 8 characters
        "name": "Test User"
    }

    try:
        response = requests.post(
            "http://localhost:8002/api/auth/signup",
            headers={"Content-Type": "application/json"},
            data=json.dumps(signup_data),
            timeout=10
        )
        print(f"   8-char password - Status: {response.status_code}")
        print(f"   Response: {response.text}")
    except Exception as e:
        print(f"   8-char password error: {e}")

    # Test with a 9-character password (should fail)
    print("\n2. Testing signup with 9-character password...")
    signup_data_9 = {
        "email": f"testuser2{int(time.time())}@example.com",
        "password": "123456789",  # 9 characters - should fail
        "name": "Test User 2"
    }

    try:
        response = requests.post(
            "http://localhost:8002/api/auth/signup",
            headers={"Content-Type": "application/json"},
            data=json.dumps(signup_data_9),
            timeout=10
        )
        print(f"   9-char password - Status: {response.status_code}")
        print(f"   Response: {response.text}")
    except Exception as e:
        print(f"   9-char password error: {e}")

    print("\nPassword validation test completed!")

if __name__ == "__main__":
    test_short_password()