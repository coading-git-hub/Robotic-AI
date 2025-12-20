#!/usr/bin/env python3
"""
Start the unified API server with proper error handling and configuration checks.
"""
import sys
import os
import uvicorn
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the backend directory to the path
backend_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, backend_dir)

# Add parent directory to path for rag_agent_api import
parent_dir = os.path.dirname(backend_dir)
sys.path.insert(0, parent_dir)

def check_configuration():
    """Check if required configuration is present."""
    print("=" * 60)
    print("Checking Configuration...")
    print("=" * 60)
    
    # Check database URL
    neon_url = os.getenv("NEON_DATABASE_URL")
    db_url = os.getenv("DATABASE_URL")
    
    if neon_url:
        print("✓ NEON_DATABASE_URL found")
        # Mask the password in the URL for security
        masked_url = neon_url.split("@")[0] + "@***" if "@" in neon_url else neon_url
        print(f"  Database: {masked_url}")
    elif db_url:
        print("✓ DATABASE_URL found")
        masked_url = db_url.split("@")[0] + "@***" if "@" in db_url else db_url
        print(f"  Database: {masked_url}")
    else:
        print("⚠ WARNING: No database URL found!")
        print("  Using SQLite fallback (not recommended for production)")
        print("  Set NEON_DATABASE_URL or DATABASE_URL in .env file")
    
    # Check Better Auth secret
    auth_secret = os.getenv("BETTER_AUTH_SECRET")
    if auth_secret:
        print("✓ BETTER_AUTH_SECRET found")
    else:
        print("⚠ WARNING: BETTER_AUTH_SECRET not set, using default")
    
    print("=" * 60)
    print()

def start_server():
    """Start the unified API server."""
    try:
        print("Starting Unified API Service on port 8002...")
        print("=" * 60)
        print("Endpoints:")
        print("  - Health: http://localhost:8002/api/health")
        print("  - RAG Query: http://localhost:8002/api/agent/query")
        print("  - Auth Signup: http://localhost:8002/api/auth/signup")
        print("  - Auth Signin: http://localhost:8002/api/auth/signin")
        print("  - Better-Auth Signup: http://localhost:8002/api/auth/sign-up/email")
        print("  - API Docs: http://localhost:8002/docs")
        print("=" * 60)
        print()
        
        # Import the app from rag_agent_api
        from rag_agent_api import app
        
        # Run the app
        uvicorn.run(app, host="0.0.0.0", port=8002, reload=False)
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("\nMake sure you're in the backend directory and all dependencies are installed:")
        print("  pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    check_configuration()
    start_server()

