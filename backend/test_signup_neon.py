#!/usr/bin/env python3
"""
Test script to verify signup functionality works with Neon database.
"""
import requests
import json
import time
import sys
import os

# Add backend directory to path for imports
backend_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, backend_dir)

def test_signup_with_neon():
    """
    Test signup functionality to ensure user data is saved to Neon database.
    """
    print("=" * 60)
    print("Testing Signup with Neon Database")
    print("=" * 60)

    # Test user data (password must be exactly 8 characters)
    test_user = {
        "email": "testuser@example.com",
        "name": "Test User",
        "password": "test1234"  # Exactly 8 characters
    }

    print(f"Test user data: {test_user}")

    # Test the signup endpoint
    signup_url = "http://localhost:8002/api/auth/signup"
    print(f"\nTesting signup endpoint: {signup_url}")

    try:
        response = requests.post(
            signup_url,
            json=test_user,
            headers={"Content-Type": "application/json"},
            timeout=30  # Longer timeout for database operations
        )

        print(f"Response Status: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")

        try:
            response_data = response.json()
            print(f"Response JSON: {json.dumps(response_data, indent=2)}")
        except:
            print(f"Response Text: {response.text}")

        if response.status_code == 200:
            print("✓ Signup successful!")

            # Extract the token for verification
            if 'session' in response_data and 'token' in response_data['session']:
                token = response_data['session']['token']
                print(f"✓ Token received: {token[:20]}...")

                # Test profile endpoint to verify user was created
                profile_url = "http://localhost:8002/api/auth/profile"
                profile_response = requests.get(
                    profile_url,
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Content-Type": "application/json"
                    },
                    timeout=30
                )

                print(f"\nProfile response status: {profile_response.status_code}")
                if profile_response.status_code == 200:
                    profile_data = profile_response.json()
                    print(f"Profile data: {json.dumps(profile_data, indent=2)}")
                    print("✓ User profile retrieved successfully - user was saved to database!")
                    return True
                else:
                    print(f"✗ Failed to retrieve profile: {profile_response.status_code}")
                    print(f"Profile response: {profile_response.text}")
                    return False
            else:
                print("✗ No token in response")
                return False
        else:
            print(f"✗ Signup failed with status {response.status_code}")
            return False

    except requests.exceptions.ConnectionError:
        print("✗ Connection error - is the server running on http://localhost:8002?")
        print("  Please start the server with: python start_server.py")
        return False
    except requests.exceptions.Timeout:
        print("✗ Request timed out - database might be slow to respond")
        return False
    except Exception as e:
        print(f"✗ Error during signup test: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_better_auth_signup():
    """
    Test the better-auth compatible signup endpoint.
    """
    print("\n" + "=" * 60)
    print("Testing Better-Auth Compatible Signup")
    print("=" * 60)

    # Test user data for better-auth endpoint
    test_user = {
        "email": "testuser2@example.com",
        "name": "Test User 2",
        "password": "test5678"  # Exactly 8 characters
    }

    print(f"Test user data: {test_user}")

    # Test the better-auth compatible endpoint
    signup_url = "http://localhost:8002/api/auth/sign-up/email"
    print(f"\nTesting better-auth endpoint: {signup_url}")

    try:
        response = requests.post(
            signup_url,
            json=test_user,
            headers={"Content-Type": "application/json"},
            timeout=30
        )

        print(f"Response Status: {response.status_code}")

        try:
            response_data = response.json()
            print(f"Response JSON: {json.dumps(response_data, indent=2)}")
        except:
            print(f"Response Text: {response.text}")

        if response.status_code == 200:
            print("✓ Better-auth signup successful!")
            return True
        else:
            print(f"✗ Better-auth signup failed with status {response.status_code}")
            return False

    except requests.exceptions.ConnectionError:
        print("✗ Connection error - is the server running on http://localhost:8002?")
        return False
    except Exception as e:
        print(f"✗ Error during better-auth signup test: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """
    Main function to run all tests.
    """
    print("Testing signup functionality with Neon database integration...")

    # Wait a moment for server to be ready if it was just started
    time.sleep(2)

    success1 = test_signup_with_neon()
    success2 = test_better_auth_signup()

    print("\n" + "=" * 60)
    print("Test Results:")
    print(f"  Standard Signup: {'PASS' if success1 else 'FAIL'}")
    print(f"  Better-Auth Signup: {'PASS' if success2 else 'FAIL'}")
    print("=" * 60)

    if success1 and success2:
        print("✓ All tests passed! Signup is working with Neon database.")
        return True
    else:
        print("✗ Some tests failed. Please check the server logs and database configuration.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)