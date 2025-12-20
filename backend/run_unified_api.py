#!/usr/bin/env python3
"""
Simple script to run the unified API on port 8002.
"""
import sys
import os
import uvicorn

# Add the backend directory to the path
backend_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, backend_dir)

# Add parent directory to path for rag_agent_api import
parent_dir = os.path.dirname(backend_dir)
sys.path.insert(0, parent_dir)

try:
    print("Starting Unified API Service on port 8002...")
    print("RAG API endpoints available at: http://localhost:8002/api/agent/*")
    print("Auth API endpoints available at: http://localhost:8002/api/auth/*")
    print("Health check available at: http://localhost:8002/api/health")

    # Import the app
    from src.main import app

    # Run the app
    uvicorn.run(app, host="0.0.0.0", port=8002, reload=False)

except Exception as e:
    print(f"Error starting unified API: {e}")
    import traceback
    traceback.print_exc()