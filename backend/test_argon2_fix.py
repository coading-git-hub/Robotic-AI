#!/usr/bin/env python3
"""
Test script to verify password hashing works correctly with Argon2.
"""
import sys
import os

# Add backend directory to path for imports
backend_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, backend_dir)

def test_password_hashing_argon2():
    """
    Test the password hashing functionality with Argon2.
    """
    print("Testing password hashing functionality with Argon2...")

    # Import the AuthService
    from src.auth.services import AuthService

    # Test cases with different types of 8-character passwords
    test_passwords = [
        "test1234",      # Standard alphanumeric
        "password",      # Standard letters
        "abc123!@",      # With special characters
        "ABCDEFGH",      # Uppercase
        "abcdefgh",      # Lowercase
        "12345678",      # Numbers only
        "test!@#$",      # Mixed with special chars
    ]

    print("\nTesting valid ASCII passwords (should work):")
    for pwd in test_passwords:
        try:
            # Test hashing
            hashed = AuthService.get_password_hash(pwd)
            print(f"[OK] Password '{pwd}' -> Hashed successfully")

            # Test verification
            is_valid = AuthService.verify_password(pwd, hashed)
            if is_valid:
                print(f"[OK] Password '{pwd}' -> Verification successful")
            else:
                print(f"[ERR] Password '{pwd}' -> Verification failed")

        except Exception as e:
            print(f"[ERR] Password '{pwd}' -> Error: {str(e)}")

    print("\nTesting the complete user creation process:")
    # Mock session for testing
    class MockSession:
        def add(self, obj): pass
        def commit(self): pass
        def refresh(self, obj): pass

    valid_user_data = {
        "email": "test@example.com",
        "password": "test1234",  # 8-character ASCII password
        "name": "Test User"
    }

    try:
        user = AuthService.create_user(
            db=MockSession(),
            email=valid_user_data["email"],
            password=valid_user_data["password"],
            name=valid_user_data["name"]
        )
        print(f"[OK] User creation successful for '{valid_user_data['email']}'")
    except Exception as e:
        print(f"[ERR] User creation failed: {str(e)}")

    print("\nAll Argon2 tests completed!")

if __name__ == "__main__":
    test_password_hashing_argon2()