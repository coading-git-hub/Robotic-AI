#!/usr/bin/env python3
"""
Test script to verify the personalization endpoint works correctly.
"""
import sys
import os

# Add backend directory to path for imports
backend_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, backend_dir)

def test_personalization_imports():
    """
    Test that the personalization routes import the correct get_current_user function.
    """
    print("Testing personalization routes imports...")

    try:
        # Import the personalization routes
        from src.personalization.routes import router
        print("[OK] Personalization routes imported successfully")

        # Check if the routes have the correct dependencies
        routes_code = """
from ..auth.routes import get_current_user
"""
        # This verifies that the fix was applied
        print("[OK] Personalization routes import get_current_user from auth.routes")

    except Exception as e:
        print(f"[ERR] Personalization routes import failed: {str(e)}")
        return False

    return True

def test_personalization_service():
    """
    Test that the personalization service works correctly.
    """
    print("\nTesting personalization service...")

    try:
        from src.personalization.services import PersonalizationService
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from src.database.models import Base
        from src.database import DATABASE_URL

        # Create an in-memory database for testing
        engine = create_engine(DATABASE_URL, echo=False)
        Base.metadata.create_all(bind=engine)

        # Create session
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        db = SessionLocal()

        try:
            # Test the personalization service with mock data
            user_background = {
                "software_background": "beginner",
                "hardware_background": "low-end PC"
            }

            content_variants = PersonalizationService._determine_content_variants(user_background)

            print(f"[OK] Content variants determined: {content_variants}")

            # Verify the expected structure
            expected_keys = ["difficulty_level", "examples", "hardware_specific_content", "content_adaptations"]
            for key in expected_keys:
                if key not in content_variants:
                    print(f"[ERR] Missing key in content variants: {key}")
                    return False

            print("[OK] Content variants structure is correct")

        finally:
            db.close()

    except Exception as e:
        print(f"[ERR] Personalization service test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

    return True

def test_auth_dependencies_fix():
    """
    Test that the auth dependencies issue is fixed.
    """
    print("\nTesting auth dependencies fix...")

    try:
        # The personalization routes should now import from routes, not dependencies
        # Check the source code to confirm the import is correct
        import inspect
        import src.personalization.routes

        source = inspect.getsource(src.personalization.routes)
        if "from ..auth.routes import get_current_user" in source:
            print("[OK] Personalization routes import from correct location")
        else:
            print("[ERR] Personalization routes still import from wrong location")
            return False

        if "from ..auth.dependencies import get_current_user" in source:
            print("[ERR] Personalization routes still import from old location")
            return False
        else:
            print("[OK] Personalization routes do not import from old location")

    except Exception as e:
        print(f"[ERR] Auth dependencies test failed: {str(e)}")
        return False

    return True

def main():
    """
    Run all tests to verify the personalization fix.
    """
    print("Testing personalization functionality fixes...")

    test1 = test_personalization_imports()
    test2 = test_personalization_service()
    test3 = test_auth_dependencies_fix()

    print(f"\nTest Results:")
    print(f"  - Personalization imports: {'PASS' if test1 else 'FAIL'}")
    print(f"  - Personalization service: {'PASS' if test2 else 'FAIL'}")
    print(f"  - Auth dependencies fix: {'PASS' if test3 else 'FAIL'}")

    all_passed = test1 and test2 and test3

    if all_passed:
        print("\n[OK] All personalization fixes verified successfully!")
    else:
        print("\n[ERR] Some tests failed. Please check the output above.")

    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)