#!/usr/bin/env python3
"""
Final test to verify the signup process works with Argon2 implementation.
"""
import sys
import os

# Add backend directory to path for imports
backend_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, backend_dir)

def test_complete_signup_flow():
    """
    Test the complete signup flow with the Argon2 implementation.
    """
    print("Testing complete signup flow with Argon2 implementation...")

    # Import the AuthService
    from src.auth.services import AuthService
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from src.database.models import Base
    from src.database import DATABASE_URL

    print(f"Using database URL: {DATABASE_URL.replace('@', '***') if '@' in DATABASE_URL else DATABASE_URL}")

    # Create an in-memory database for testing (or use the configured one)
    engine = create_engine(DATABASE_URL, echo=False)

    # Create tables
    Base.metadata.create_all(bind=engine)

    # Create session
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = SessionLocal()

    try:
        # Test user data
        test_user = {
            "email": "finaltest@example.com",
            "password": "test1234",  # 8-character ASCII password
            "name": "Final Test User"
        }

        print(f"\nTesting user creation with: {test_user}")

        # Test user creation (signup)
        user = AuthService.create_user(
            db=db,
            email=test_user["email"],
            password=test_user["password"],
            name=test_user["name"]
        )

        print(f"[OK] User created successfully:")
        print(f"  - ID: {user.id}")
        print(f"  - Email: {user.email}")
        print(f"  - Name: {user.name}")

        # Test authentication (signin)
        authenticated_user = AuthService.authenticate_user(
            db=db,
            email=test_user["email"],
            password=test_user["password"]
        )

        if authenticated_user:
            print(f"[OK] User authentication successful for '{test_user['email']}'")
        else:
            print(f"[ERR] User authentication failed for '{test_user['email']}'")

        # Test with wrong password
        wrong_auth = AuthService.authenticate_user(
            db=db,
            email=test_user["email"],
            password="wrongpassword"
        )

        if not wrong_auth:
            print(f"[OK] Authentication correctly failed with wrong password")
        else:
            print(f"[ERR] Authentication should have failed with wrong password")

        # Test with invalid characters in password (should be filtered)
        try:
            invalid_password_user = AuthService.create_user(
                db=db,
                email="invalidtest@example.com",
                password="test123!",  # Should work - all ASCII
                name="Invalid Test User"
            )
            print(f"[OK] User with special character created successfully")
        except Exception as e:
            print(f"[ERR] Failed to create user with special character: {str(e)}")

    except Exception as e:
        print(f"[ERR] Error during signup flow test: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # Close the database session
        db.close()

    print("\nFinal signup flow test completed!")

if __name__ == "__main__":
    test_complete_signup_flow()