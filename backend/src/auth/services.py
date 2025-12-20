"""
Authentication services for user registration, login, and profile management.
"""
from sqlalchemy.orm import Session
from ..database.models import User
from passlib.context import CryptContext
import uuid
from datetime import datetime
from typing import Optional

# Password hashing context using Argon2 which is more secure and doesn't have bcrypt backend issues
pwd_context = CryptContext(
    schemes=["argon2"],  # Use Argon2 instead of bcrypt
    deprecated="auto",
    argon2__rounds=10,    # Number of iterations
    argon2__memory_cost=102400,  # Memory cost in KB (100MB)
    argon2__parallelism=1  # Number of parallel threads
)

# Initialize argon2 to ensure it's available
try:
    import argon2
    # Test argon2 functionality
    ph = argon2.PasswordHasher()
    test_hash = ph.hash("test1234")
    ph.verify(test_hash, "test1234")
except ImportError:
    # Handle case where argon2 is not available
    pass
except Exception:
    # Handle any other argon2 initialization issues
    pass

def truncate_to_bytes(password: str, max_bytes: int = 72) -> str:
    """
    Safely truncate a password string to ensure its UTF-8 encoding doesn't exceed max_bytes.
    This handles multi-byte characters correctly.
    """
    password_bytes = password.encode('utf-8')
    if len(password_bytes) <= max_bytes:
        return password
    
    # Truncate bytes and decode, handling potential incomplete characters
    truncated_bytes = password_bytes[:max_bytes]
    # Remove any incomplete trailing bytes (UTF-8 continuation bytes start with 10xxxxxx)
    while truncated_bytes and (truncated_bytes[-1] & 0xC0) == 0x80:
        truncated_bytes = truncated_bytes[:-1]
    
    return truncated_bytes.decode('utf-8', errors='ignore')

class AuthService:
    """
    Service class for authentication operations.
    """

    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """
        Verify a plain password against a hashed password.
        Uses Argon2 password verification which is secure and reliable.
        """
        # Truncate password to 8 characters to match the stored hash
        truncated_password = plain_password[:8] if len(plain_password) > 8 else plain_password

        # Ensure it's ASCII and compatible with password hashing requirements
        # Only allow standard ASCII characters (32-126 are printable ASCII)
        ascii_password = ''.join(c for c in truncated_password if 32 <= ord(c) <= 126)  # Printable ASCII range

        return pwd_context.verify(ascii_password, hashed_password)

    @staticmethod
    def get_password_hash(password: str) -> str:
        """
        Generate a hash for a plain password.
        Uses Argon2 password hashing algorithm which is secure and doesn't have the same limitations as bcrypt.
        """
        # First truncate to 8 characters to meet our business requirements
        truncated_password = password[:8] if len(password) > 8 else password

        # Ensure it's ASCII and compatible with password hashing requirements
        # Only allow standard ASCII characters (32-126 are printable ASCII)
        ascii_password = ''.join(c for c in truncated_password if 32 <= ord(c) <= 126)  # Printable ASCII range

        # Final safety check - ensure the password is properly formatted
        # Since we're limiting to 8 characters and only printable ASCII, this should be fine
        password_bytes = ascii_password.encode('utf-8')
        if len(password_bytes) > 72:
            # This should never happen if we're only using 8 characters, but just in case
            ascii_password = ascii_password[:72]
            password_bytes = ascii_password.encode('utf-8')

        try:
            # Make sure we're sending a properly formatted string to the password hasher
            if not ascii_password:
                raise ValueError("Password cannot be empty after ASCII filtering")
            return pwd_context.hash(ascii_password)
        except ValueError as e:
            if "72 bytes" in str(e).lower() or "truncate" in str(e).lower():
                # If we still get this error, there's an issue with the password hashing backend
                raise ValueError(f"Password encoding error: {str(e)}. Please use only standard ASCII characters (letters, numbers, and basic symbols).")
            raise
        except Exception as e:
            # Catch any other password hashing-related errors
            raise ValueError(f"Password hashing error: {str(e)}. Please ensure your password contains only standard ASCII characters.")

    @classmethod
    def create_user(
        cls,
        db: Session,
        email: str,
        password: str,
        name: Optional[str] = None,
        software_background: Optional[str] = None,
        hardware_background: Optional[str] = None
    ) -> User:
        """
        Create a new user with background information.
        """
        try:
            # Check if user already exists
            existing_user = db.query(User).filter(User.email == email).first()
            if existing_user:
                raise ValueError("Email already registered")

            # Validate password length (exactly 8 characters required)
            if len(password) != 8:
                raise ValueError("Password must be exactly 8 characters")

            # Check that all characters are printable ASCII (32-126)
            # This prevents issues with password hashing and encoding problems
            for char in password:
                if ord(char) < 32 or ord(char) > 126:
                    raise ValueError("Password contains invalid characters. Please use only standard ASCII characters (letters, numbers, and basic symbols).")

            # Create password hash
            hashed_password = cls.get_password_hash(password)

            # Create new user
            user = User(
                id=str(uuid.uuid4()),
                email=email,
                name=name,
                hashed_password=hashed_password,  # Use the already hashed password
                software_background=software_background,
                hardware_background=hardware_background,
                background_updated_at=datetime.utcnow() if software_background or hardware_background else None
            )

            db.add(user)
            db.commit()
            db.refresh(user)

            return user
        except ValueError:
            # Re-raise ValueError as-is (validation errors)
            raise
        except Exception as e:
            # Log the error for debugging
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Error creating user: {str(e)}")
            # Rollback the transaction in case of error
            db.rollback()
            raise ValueError(f"Failed to create user: {str(e)}")

    @classmethod
    def authenticate_user(cls, db: Session, email: str, password: str) -> Optional[User]:
        """
        Authenticate a user with email and password.
        """
        user = db.query(User).filter(User.email == email).first()
        if not user or not cls.verify_password(password, user.hashed_password):
            return None
        return user

    @classmethod
    def get_user_by_email(cls, db: Session, email: str) -> Optional[User]:
        """
        Get a user by email.
        """
        return db.query(User).filter(User.email == email).first()

    @classmethod
    def get_user_by_id(cls, db: Session, user_id: str) -> Optional[User]:
        """
        Get a user by ID.
        """
        return db.query(User).filter(User.id == user_id).first()

    @classmethod
    def update_user_background(
        cls,
        db: Session,
        user_id: str,
        software_background: Optional[str] = None,
        hardware_background: Optional[str] = None
    ) -> Optional[User]:
        """
        Update user's background information.
        """
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            return None

        if software_background is not None:
            user.software_background = software_background
        if hardware_background is not None:
            user.hardware_background = hardware_background

        user.background_updated_at = datetime.utcnow()
        user.updated_at = datetime.utcnow()

        db.commit()
        db.refresh(user)

        return user

    @classmethod
    def update_user_profile(
        cls,
        db: Session,
        user_id: str,
        name: Optional[str] = None,
        software_background: Optional[str] = None,
        hardware_background: Optional[str] = None
    ) -> Optional[User]:
        """
        Update user's profile information including name and background.
        """
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            return None

        # Update name if provided
        if name is not None:
            user.name = name

        # Update background information if provided
        if software_background is not None:
            user.software_background = software_background
        if hardware_background is not None:
            user.hardware_background = hardware_background

        # Update the background_updated_at if background fields are being updated
        if software_background is not None or hardware_background is not None:
            user.background_updated_at = datetime.utcnow()

        user.updated_at = datetime.utcnow()

        db.commit()
        db.refresh(user)

        return user