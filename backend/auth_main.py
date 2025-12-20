"""
Main FastAPI application for the authentication API.
This serves the authentication endpoints that the frontend expects.
"""
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from src.database import create_tables, get_db
from src.auth.routes import router as auth_routes
from src.personalization import routes as personalization_routes
import os
from dotenv import load_dotenv

load_dotenv()

# Create tables on startup
create_tables()

# Create FastAPI app instance
app = FastAPI(
    title="Authentication API",
    description="API for user authentication with background collection and content personalization",
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

# Include custom auth routes (these match what the frontend expects)
app.include_router(auth_routes, prefix="/api/auth", tags=["Authentication"])

# Include personalization routes
app.include_router(personalization_routes.router, prefix="/api/personalization", tags=["Personalization"])

@app.get("/")
def read_root():
    return {"message": "Authentication API - Provides signup, signin, and profile endpoints"}

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "auth-api"}

# Database dependency
def get_database():
    db = next(get_db())
    try:
        yield db
    finally:
        db.close()

if __name__ == "__main__":
    import uvicorn
    print("Starting Authentication API on port 8001...")
    print("API Documentation available at: http://localhost:8001/docs")
    print("Signup endpoint: http://localhost:8001/api/auth/signup")
    print("Signin endpoint: http://localhost:8001/api/auth/signin")
    print("Profile endpoint: http://localhost:8001/api/auth/profile")
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=False)