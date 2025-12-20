#!/usr/bin/env python3
"""
Final test of auth endpoints to confirm they're working.
"""
import requests
import time
import json

def test_auth_endpoints():
    print("Testing auth endpoints specifically...")

    # Wait a moment for server stability
    time.sleep(1)

    print("\n1. Testing signup endpoint...")
    signup_data = {
        "email": f"testuser{int(time.time())}@example.com",  # Use timestamp to ensure unique email
        "password": "shortpass123",  # Under 72 bytes for bcrypt
        "name": "Test User"
    }

    try:
        response = requests.post(
            "http://localhost:8002/api/auth/signup",
            headers={"Content-Type": "application/json"},
            data=json.dumps(signup_data),
            timeout=10
        )
        print(f"   Signup endpoint - Status: {response.status_code}")
        print(f"   Response: {response.text}")

        # If signup was successful, try to get the token from response
        if response.status_code == 200:
            print("   [SUCCESS] Signup successful!")
        elif response.status_code == 400:
            print("   [ERROR] Signup failed with validation error (likely duplicate email)")
        else:
            print(f"   [ERROR] Signup failed with status {response.status_code}")
    except Exception as e:
        print(f"   [ERROR] Signup endpoint error: {e}")

    print("\n2. Testing signin endpoint...")
    signin_data = {
        "email": signup_data["email"],  # Use the same email we tried to register
        "password": "shortpass123"
    }

    try:
        response = requests.post(
            "http://localhost:8002/api/auth/signin",
            headers={"Content-Type": "application/json"},
            data=json.dumps(signin_data),
            timeout=10
        )
        print(f"   Signin endpoint - Status: {response.status_code}")
        print(f"   Response: {response.text}")

        if response.status_code == 200:
            print("   [SUCCESS] Signin successful!")
        elif response.status_code == 401:
            print("   [ERROR] Signin failed - user may not exist yet")
        else:
            print(f"   [ERROR] Signin failed with status {response.status_code}")
    except Exception as e:
        print(f"   [ERROR] Signin endpoint error: {e}")

    print("\n3. Testing profile endpoint (should require auth)...")
    try:
        response = requests.get(
            "http://localhost:8002/api/auth/profile",
            timeout=10
        )
        print(f"   Profile endpoint - Status: {response.status_code}")
        print(f"   Response: {response.text}")
    except Exception as e:
        print(f"   [ERROR] Profile endpoint error: {e}")

    print("\nAuth endpoint tests completed!")

if __name__ == "__main__":
    test_auth_endpoints()