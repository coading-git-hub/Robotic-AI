import uvicorn
import sys
import os

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the app from rag_agent_api
from rag_agent_api import app

if __name__ == "__main__":
    print("Starting RAG Agent API on port 8001...")
    print("API Documentation available at: http://localhost:8001/docs")
    print("Health check available at: http://localhost:8001/api/health")
    print("Query endpoint available at: http://localhost:8001/api/agent/query")
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=False)