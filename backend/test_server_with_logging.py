#!/usr/bin/env python3
"""
Test server startup with detailed logging to see what happens.
"""
import uvicorn
import sys
import os

# Add paths
backend_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, backend_dir)
parent_dir = os.path.dirname(backend_dir)
sys.path.insert(0, parent_dir)

def run_server():
    print("Starting server with detailed logging...")

    # Import and create app to check routes
    from src.main import app
    print("Routes registered:")
    for route in app.routes:
        if hasattr(route, 'methods') and hasattr(route, 'path'):
            methods = ', '.join(sorted(route.methods))
            print(f"  {methods} {route.path}")

    print("\nStarting server...")
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8002,
        reload=False,
        log_level="debug"  # Enable debug logging
    )

if __name__ == "__main__":
    run_server()