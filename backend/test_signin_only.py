#!/usr/bin/env python3
"""
Test the signin endpoint to see if it avoids the bcrypt initialization issue.
"""
import requests
import time
import json

def test_signin():
    print("Testing signin endpoint...")

    # Test with a 4-character password (should fail because user doesn't exist)
    print("\n1. Testing signin with 4-character password...")
    signin_data = {
        "email": "nonexistent@example.com",
        "password": "1234",  # 4 characters
    }

    try:
        response = requests.post(
            "http://localhost:8002/api/auth/signin",
            headers={"Content-Type": "application/json"},
            data=json.dumps(signin_data),
            timeout=10
        )
        print(f"   Signin - Status: {response.status_code}")
        print(f"   Response: {response.text}")
    except Exception as e:
        print(f"   Signin error: {e}")

    print("\nSignin test completed!")

if __name__ == "__main__":
    test_signin()