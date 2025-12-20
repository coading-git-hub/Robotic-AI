#!/usr/bin/env python3
"""
Simple test to check bcrypt functionality
"""
import sys
import os

# Add backend directory to path for imports
backend_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, backend_dir)

def test_bcrypt_directly():
    """
    Test bcrypt directly to see if there are issues with the library
    """
    print("Testing bcrypt directly...")

    try:
        import bcrypt
        print("✓ bcrypt imported successfully")

        # Test with a simple 8-character ASCII password
        password = "test1234"
        print(f"Testing with password: '{password}'")

        # Convert to bytes for bcrypt
        password_bytes = password.encode('utf-8')
        print(f"Password as bytes: {password_bytes}")
        print(f"Password byte length: {len(password_bytes)}")

        # Generate salt and hash
        salt = bcrypt.gensalt()
        print(f"Salt generated: {salt}")

        # Hash the password
        hashed = bcrypt.hashpw(password_bytes, salt)
        print(f"Password hashed successfully: {hashed}")

        # Verify the password
        is_valid = bcrypt.checkpw(password_bytes, hashed)
        print(f"Password verification: {is_valid}")

        if is_valid:
            print("✓ Direct bcrypt test passed!")
        else:
            print("✗ Direct bcrypt test failed - verification returned False")

    except ImportError as e:
        print(f"✗ bcrypt import failed: {e}")
    except Exception as e:
        print(f"✗ bcrypt test failed with error: {e}")
        import traceback
        traceback.print_exc()

def test_passlib_directly():
    """
    Test passlib directly to see if there are issues with the library
    """
    print("\nTesting passlib directly...")

    try:
        from passlib.context import CryptContext
        import passlib
        print("✓ passlib imported successfully")

        # Create a simple context
        pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        print("✓ CryptContext created")

        # Test with a simple 8-character ASCII password
        password = "test1234"
        print(f"Testing with password: '{password}'")

        # Hash the password
        hashed = pwd_context.hash(password)
        print(f"Password hashed successfully: {hashed[:30]}...")

        # Verify the password
        is_valid = pwd_context.verify(password, hashed)
        print(f"Password verification: {is_valid}")

        if is_valid:
            print("✓ Direct passlib test passed!")
        else:
            print("✗ Direct passlib test failed - verification returned False")

    except Exception as e:
        print(f"✗ passlib test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_bcrypt_directly()
    test_passlib_directly()