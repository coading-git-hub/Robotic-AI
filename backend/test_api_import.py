#!/usr/bin/env python3
"""
Test script to check if the unified API can be imported without errors.
"""
import sys
import os

# Add the backend directory to the path
backend_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, backend_dir)

# Add parent directory to path for rag_agent_api import
parent_dir = os.path.dirname(backend_dir)
sys.path.insert(0, parent_dir)

try:
    print("Attempting to import the unified API...")
    from src.main import app
    print("[SUCCESS] Successfully imported unified API")

    # Check available routes
    print("\nAvailable routes:")
    for route in app.routes:
        if hasattr(route, 'methods') and hasattr(route, 'path'):
            methods = ', '.join(sorted(route.methods))
            print(f"  {methods} {route.path}")

    print("\n[SUCCESS] API import test completed successfully")

except Exception as e:
    print(f"[ERROR] Error importing unified API: {e}")
    import traceback
    traceback.print_exc()