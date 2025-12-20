#!/usr/bin/env python3
"""
Test the signup endpoint with a proper POST request.
"""
import requests
import json

def test_signup_endpoint():
    print("Testing signup endpoint with POST request...")

    # Test data for signup
    signup_data = {
        "email": "test@example.com",
        "name": "Test User",
        "password": "testpassword123"
    }

    try:
        response = requests.post(
            "http://localhost:8002/api/auth/signup",
            json=signup_data,
            timeout=10
        )
        print(f"[SUCCESS] Signup endpoint - Status: {response.status_code}")
        print(f"Response: {response.text}")
        return True
    except requests.exceptions.ConnectionError:
        print("[ERROR] Signup endpoint - Connection refused (server not running)")
        return False
    except Exception as e:
        print(f"[ERROR] Signup endpoint - Error: {str(e)}")
        return False

def test_signin_endpoint():
    print("\nTesting signin endpoint with POST request...")

    # Test data for signin
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
        print(f"[SUCCESS] Signin endpoint - Status: {response.status_code}")
        print(f"Response: {response.text}")
        return True
    except requests.exceptions.ConnectionError:
        print("[ERROR] Signin endpoint - Connection refused (server not running)")
        return False
    except Exception as e:
        print(f"[ERROR] Signin endpoint - Error: {str(e)}")
        return False

if __name__ == "__main__":
    print("Testing unified API auth endpoints...")

    signup_ok = test_signup_endpoint()
    if signup_ok:
        test_signin_endpoint()