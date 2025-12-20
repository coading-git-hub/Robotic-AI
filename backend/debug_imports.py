#!/usr/bin/env python3
"""
Debug import issues with the unified API.
"""
import sys
import os

# Add paths
backend_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, backend_dir)
parent_dir = os.path.dirname(backend_dir)
sys.path.insert(0, parent_dir)

def test_imports():
    print("Testing individual imports...")

    # Test basic imports
    try:
        print("1. Testing database import...")
        from src.database import create_tables, get_db
        print("   [SUCCESS] Database import successful")
    except Exception as e:
        print(f"   [ERROR] Database import failed: {e}")
        import traceback
        traceback.print_exc()

    try:
        print("2. Testing auth routes import...")
        from src.auth.routes import router as auth_routes
        print("   [SUCCESS] Auth routes import successful")
        print(f"   Router: {auth_routes}")
        print(f"   Routes in auth router: {[route.path for route in auth_routes.routes]}")
    except Exception as e:
        print(f"   [ERROR] Auth routes import failed: {e}")
        import traceback
        traceback.print_exc()

    try:
        print("3. Testing personalization routes import...")
        from src.personalization import routes as personalization_routes
        print("   [SUCCESS] Personalization routes import successful")
        print(f"   Router: {personalization_routes.router}")
        print(f"   Routes in personalization router: {[route.path for route in personalization_routes.router.routes]}")
    except Exception as e:
        print(f"   [ERROR] Personalization routes import failed: {e}")
        import traceback
        traceback.print_exc()

    try:
        print("4. Testing rag_agent_api import...")
        from rag_agent_api import AgentQueryRequest, AgentQueryResponse
        print("   [SUCCESS] RAG agent API import successful")
    except Exception as e:
        print(f"   [ERROR] RAG agent API import failed: {e}")
        import traceback
        traceback.print_exc()

    # Test creating the app
    try:
        print("5. Testing app creation...")
        from src.main import app
        print("   [SUCCESS] App creation successful")
        print(f"   Total routes: {len(app.routes)}")
        for route in app.routes:
            if hasattr(route, 'methods') and hasattr(route, 'path'):
                methods = ', '.join(sorted(route.methods))
                print(f"     {methods} {route.path}")
    except Exception as e:
        print(f"   [ERROR] App creation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_imports()