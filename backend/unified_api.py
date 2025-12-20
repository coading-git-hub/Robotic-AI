"""
Main FastAPI application for both RAG and Authentication APIs.
This serves both the RAG API and authentication endpoints on the same port.
"""
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from src.database import create_tables, get_db
from src.auth.routes import router as auth_routes
from src.personalization import routes as personalization_routes
import os
from dotenv import load_dotenv

# Import the RAG agent app components
from rag_agent_api import (
    app as rag_app,
    router as rag_router,
    # Import other components from rag_agent_api
)

load_dotenv()

# Create tables on startup
create_tables()

# Create FastAPI app instance
app = FastAPI(
    title="Unified API Service",
    description="API for both RAG functionality and user authentication",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include RAG agent routes under /api prefix to match frontend expectations
app.include_router(rag_router, prefix="/api/agent", tags=["RAG Agent"])

# Include auth routes
app.include_router(auth_routes, prefix="/api/auth", tags=["Authentication"])

# Include personalization routes
app.include_router(personalization_routes.router, prefix="/api/personalization", tags=["Personalization"])

@app.get("/")
def read_root():
    return {"message": "Unified API Service - RAG and Authentication"}

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "unified-api"}

# Database dependency
def get_database():
    db = next(get_db())
    try:
        yield db
    finally:
        db.close()

if __name__ == "__main__":
    import uvicorn
    print("Starting Unified API Service on port 8002...")
    print("RAG API endpoints available at: http://localhost:8002/api/agent/*")
    print("Auth API endpoints available at: http://localhost:8002/api/auth/*")
    print("Health check available at: http://localhost:8002/health")
    uvicorn.run(app, host="0.0.0.0", port=8002, reload=False)