"""
Database models for the authentication and personalization system.

This module defines the database schema using SQLAlchemy models.
"""
from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
import uuid

Base = declarative_base()

class User(Base):
    """
    User model extending the default authentication user with background information.
    """
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String, unique=True, nullable=False)
    name = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Password field
    hashed_password = Column(String, nullable=False)

    # Background information fields
    software_background = Column(String(50), nullable=True)  # Beginner, Frontend, Backend, AI
    hardware_background = Column(String(50), nullable=True)  # Low-end PC, Mid-range, High-end, GPU
    background_updated_at = Column(DateTime, nullable=True)

    def __repr__(self):
        return f"<User(email='{self.email}', software_background='{self.software_background}', hardware_background='{self.hardware_background}')>"

class PersonalizationSession(Base):
    """
    Model to track personalization sessions when users click the personalize button.
    """
    __tablename__ = "personalization_sessions"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, nullable=False)  # Reference to user
    chapter_id = Column(String, nullable=False)  # ID of the chapter being personalized
    background_applied = Column(JSON, nullable=False)  # The user's background data used for personalization
    personalization_applied = Column(Boolean, default=False)  # Whether personalization was successfully applied
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<PersonalizationSession(user_id='{self.user_id}', chapter_id='{self.chapter_id}')>"