#!/usr/bin/env python3
"""
Debug password hashing to understand where the bcrypt error is coming from.
"""
import sys
import os

# Add paths
backend_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, backend_dir)
parent_dir = os.path.dirname(backend_dir)
sys.path.insert(0, parent_dir)

def test_bcrypt_directly():
    print("Testing bcrypt directly with different password lengths...")

    from passlib.context import CryptContext
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

    # Test various password lengths
    test_passwords = [
        "123456",    # 6 chars
        "1234567",   # 7 chars
        "12345678",  # 8 chars
        "123456789", # 9 chars
        "1234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890"  # 110 chars
    ]

    for i, pwd in enumerate(test_passwords):
        try:
            print(f"\nTest {i+1}: Password length {len(pwd)}")
            print(f"Password: '{pwd}'")
            hashed = pwd_context.hash(pwd)
            print(f"✅ Hashing successful")
        except Exception as e:
            print(f"❌ Hashing failed: {e}")

def test_validation_logic():
    print("\n\nTesting our validation logic...")

    # Import our AuthService
    from src.auth.services import AuthService

    test_passwords = [
        "123456",    # 6 chars
        "1234567",   # 7 chars
        "12345678",  # 8 chars
        "123456789", # 9 chars
    ]

    for pwd in test_passwords:
        print(f"\nTesting password: '{pwd}' (length: {len(pwd)})")
        try:
            # Test the get_password_hash method
            hashed = AuthService.get_password_hash(pwd)
            print(f"✅ get_password_hash successful")
        except Exception as e:
            print(f"❌ get_password_hash failed: {e}")

        # Test the validation in create_user (simulated)
        try:
            if len(pwd) > 8:
                raise ValueError("Password must be 8 characters or fewer")
            print(f"✅ Length validation passed")
        except Exception as e:
            print(f"❌ Length validation failed: {e}")

if __name__ == "__main__":
    test_bcrypt_directly()
    test_validation_logic()