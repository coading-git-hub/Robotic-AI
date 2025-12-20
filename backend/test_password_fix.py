#!/usr/bin/env python3
"""
Test script to verify password hashing works correctly with the new implementation.
"""
import sys
import os

# Add backend directory to path for imports
backend_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, backend_dir)

def test_password_hashing():
    """
    Test the password hashing functionality with various inputs.
    """
    print("Testing password hashing functionality...")

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
            print(f"✓ Password '{pwd}' -> Hashed successfully")

            # Test verification
            is_valid = AuthService.verify_password(pwd, hashed)
            if is_valid:
                print(f"✓ Password '{pwd}' -> Verification successful")
            else:
                print(f"✗ Password '{pwd}' -> Verification failed")

        except Exception as e:
            print(f"✗ Password '{pwd}' -> Error: {str(e)}")

    # Test invalid passwords (non-ASCII characters)
    invalid_passwords = [
        "passwörd",     # Contains ö (non-ASCII)
        "tëst1234",     # Contains ë (non-ASCII)
        "pass字123",     # Contains Chinese character
    ]

    print("\nTesting invalid passwords (should fail validation):")
    for pwd in invalid_passwords:
        try:
            # Test creation (this should fail validation)
            from sqlalchemy.orm import Session  # Mock session for testing
            class MockSession:
                def add(self, obj): pass
                def commit(self): pass
                def refresh(self, obj): pass

            # This should raise an error due to non-ASCII characters
            try:
                AuthService.create_user(
                    db=MockSession(),
                    email="test@example.com",
                    password=pwd,
                    name="Test User"
                )
                print(f"✗ Password '{pwd}' -> Unexpectedly succeeded (should have failed)")
            except ValueError as ve:
                print(f"✓ Password '{pwd}' -> Correctly rejected: {str(ve)}")
            except Exception as e:
                print(f"✗ Password '{pwd}' -> Unexpected error: {str(e)}")

        except Exception as e:
            print(f"✗ Password '{pwd}' -> Error during test: {str(e)}")

    print("\nTesting password length validation:")
    invalid_length_passwords = [
        "short",        # Too short
        "verylongpassword",  # Too long
    ]

    for pwd in invalid_length_passwords:
        try:
            # Test creation (this should fail due to length)
            from sqlalchemy.orm import Session  # Mock session for testing
            class MockSession:
                def add(self, obj): pass
                def commit(self): pass
                def refresh(self, obj): pass

            try:
                AuthService.create_user(
                    db=MockSession(),
                    email="test@example.com",
                    password=pwd,
                    name="Test User"
                )
                print(f"✗ Password '{pwd}' -> Unexpectedly succeeded (should have failed)")
            except ValueError as ve:
                print(f"✓ Password '{pwd}' -> Correctly rejected: {str(ve)}")
            except Exception as e:
                print(f"✗ Password '{pwd}' -> Unexpected error: {str(e)}")

        except Exception as e:
            print(f"✗ Password '{pwd}' -> Error during test: {str(e)}")

    print("\nAll tests completed!")

if __name__ == "__main__":
    test_password_hashing()