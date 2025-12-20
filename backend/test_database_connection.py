#!/usr/bin/env python3
"""
Test script to verify database connection and user creation.
"""
import sys
import os

# Add backend directory to path for imports
backend_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, backend_dir)

def test_database_connection():
    """
    Test the database connection and user creation.
    """
    print("Testing database connection...")

    # Import required modules
    from sqlalchemy import create_engine, text
    from sqlalchemy.orm import sessionmaker
    from src.database import DATABASE_URL
    from src.database.models import Base, User
    from src.auth.services import AuthService

    print(f"Using database URL: {DATABASE_URL.replace('@', '***') if '@' in DATABASE_URL else DATABASE_URL}")

    try:
        # Create engine and test connection
        engine = create_engine(DATABASE_URL, echo=False)

        # Test basic connection
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            print("[OK] Basic database connection successful")

        # Create tables
        Base.metadata.create_all(bind=engine)
        print("[OK] Tables created successfully")

        # Create session
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        db = SessionLocal()

        try:
            # Test user creation
            print("\nTesting user creation...")
            test_user = AuthService.create_user(
                db=db,
                email="test_db@example.com",
                password="test1234",  # 8-character password
                name="Test User DB"
            )

            print(f"[OK] User created successfully:")
            print(f"  - ID: {test_user.id}")
            print(f"  - Email: {test_user.email}")
            print(f"  - Name: {test_user.name}")

            # Verify user exists in database
            retrieved_user = db.query(User).filter(User.email == "test_db@example.com").first()
            if retrieved_user:
                print(f"[OK] User found in database: {retrieved_user.email}")
            else:
                print("[ERR] User not found in database after creation")

            # Test authentication
            authenticated_user = AuthService.authenticate_user(
                db=db,
                email="test_db@example.com",
                password="test1234"
            )

            if authenticated_user:
                print(f"[OK] User authentication successful: {authenticated_user.email}")
            else:
                print("[ERR] User authentication failed")

        finally:
            # Close the database session
            db.close()

        print("\n[OK] Database connection and user creation test completed successfully!")

    except Exception as e:
        print(f"[ERR] Database connection test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

    return True

def check_environment():
    """
    Check if the environment variables are properly set.
    """
    print("\nChecking environment variables...")

    import os
    neon_url = os.getenv("NEON_DATABASE_URL")
    db_url = os.getenv("DATABASE_URL")

    if neon_url:
        print(f"[OK] NEON_DATABASE_URL is set: {neon_url.replace('@', '***') if '@' in neon_url else neon_url}")
    else:
        print("[INFO] NEON_DATABASE_URL is not set")

    if db_url:
        print(f"[INFO] DATABASE_URL is set: {db_url.replace('@', '***') if '@' in db_url else db_url}")
    else:
        print("[INFO] DATABASE_URL is not set")

    if not neon_url and not db_url:
        print("[WARN] Neither NEON_DATABASE_URL nor DATABASE_URL is set - will use SQLite fallback")

if __name__ == "__main__":
    check_environment()
    success = test_database_connection()

    if success:
        print("\n[OK] All database tests passed!")
    else:
        print("\n[ERR] Database tests failed!")
        sys.exit(1)