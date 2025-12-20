"""
Better Auth setup for the application.
"""
from better_auth import BaseUser, Auth
from typing import Optional
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

# Define custom user model with background fields
class ExtendedUser(BaseUser):
    software_background: Optional[str] = None
    hardware_background: Optional[str] = None
    background_updated_at: Optional[datetime] = None

# Initialize Better Auth
auth = Auth(
    secret=os.getenv("BETTER_AUTH_SECRET", "VBq3LTV3RMARjnHiqaht8zXB0hoSYk5Q"),
    user_model=ExtendedUser,
    database_url=os.getenv("NEON_DATABASE_URL", os.getenv("DATABASE_URL")),
    # Add any additional configuration here
)

# Get the router
router = auth.router