from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer
import uvicorn
import os
from dotenv import load_dotenv

# Import middleware
from middleware.security import SecurityHeadersMiddleware, RequestLoggingMiddleware, InputValidationMiddleware

# Try to import rate limiting, but make it optional
try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
    RATE_LIMITING_AVAILABLE = True
except ImportError:
    RATE_LIMITING_AVAILABLE = False
    print("Rate limiting not available - slowapi not installed")

# Load environment variables
load_dotenv()

# Import API routers
from api.routers import auth, chat, search, progress, health, indexing, content_management, assessment

# Import database setup
from db.database import engine, Base
from db.session import get_db
from core.config import settings

# Create the main FastAPI application
app = FastAPI(
    title="Physical AI & Humanoid Robotics RAG API",
    description="API for the Physical AI & Humanoid Robotics book with integrated RAG chatbot",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure rate limiter if available
if RATE_LIMITING_AVAILABLE:
    limiter = Limiter(key_func=get_remote_address)
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
else:
    print("Rate limiting middleware not loaded")

# Add custom middleware in the order they should be applied
app.add_middleware(InputValidationMiddleware)
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(SecurityHeadersMiddleware)

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    # Additional security headers can be added here
)

# Include API routers
app.include_router(auth.router, prefix="/auth", tags=["Authentication"])
app.include_router(chat.router, prefix="/chat", tags=["Chat & RAG"])
app.include_router(search.router, prefix="/search", tags=["Search"])
app.include_router(progress.router, prefix="/progress", tags=["Progress Tracking"])
app.include_router(health.router, prefix="/health", tags=["Health"])
app.include_router(indexing.router, prefix="/index", tags=["Content Indexing"])
app.include_router(content_management.router, prefix="/content", tags=["Content Management"])
app.include_router(assessment.router, prefix="/assessment", tags=["Assessment"])

# Create database tables
@app.on_event("startup")
async def startup_event():
    """Create database tables on startup"""
    async with engine.begin() as conn:
        # Create all tables defined in models
        await conn.run_sync(Base.metadata.create_all)

@app.get("/")
async def root():
    """Root endpoint for basic health check"""
    return {
        "message": "Physical AI & Humanoid Robotics RAG API",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Physical AI & Humanoid Robotics RAG API",
        "version": "1.0.0"
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )