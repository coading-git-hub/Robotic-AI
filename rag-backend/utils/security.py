from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import jwt
from passlib.context import CryptContext
from core.config import settings

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT algorithms
ALGORITHM = settings.ALGORITHM
SECRET_KEY = settings.SECRET_KEY

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plain password against a hashed password"""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Generate a hash for a plain password"""
    return pwd_context.hash(password)

def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """Create an access token with expiration"""
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode.update({"exp": expire, "type": "access"})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def create_refresh_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """Create a refresh token with longer expiration"""
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)

    to_encode.update({"exp": expire, "type": "refresh"})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str) -> Optional[Dict[str, Any]]:
    """Verify and decode a JWT token"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.PyJWTError:
        return None


# Input validation and sanitization functions
import re
import html


def sanitize_input(input_str: str) -> str:
    """Sanitize user input to prevent XSS and injection attacks"""
    if not input_str:
        return input_str

    # HTML escape the input
    sanitized = html.escape(input_str)
    return sanitized


def validate_email(email: str) -> bool:
    """Validate email format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None


def validate_content_length(content: str, max_length: int = 10000) -> bool:
    """Validate content length"""
    return len(content) <= max_length


def is_safe_content(content: str) -> bool:
    """Check if content contains potentially dangerous patterns"""
    dangerous_patterns = [
        r'<script',  # Script tags
        r'javascript:',  # JavaScript URLs
        r'on\w+\s*=',  # Event handlers
        r'<iframe',  # Iframe tags
        r'<object',  # Object tags
        r'<embed',  # Embed tags
    ]

    for pattern in dangerous_patterns:
        if re.search(pattern, content, re.IGNORECASE):
            return False

    return True


def validate_user_input(input_data: str, field_name: str = "input") -> str:
    """Validate and sanitize user input"""
    if not input_data:
        raise ValueError(f"{field_name} cannot be empty")

    # Check for dangerous content
    if not is_safe_content(input_data):
        raise ValueError(f"{field_name} contains potentially dangerous content")

    # Sanitize the input
    sanitized = sanitize_input(input_data)

    # Validate length
    if not validate_content_length(sanitized):
        raise ValueError(f"{field_name} exceeds maximum length")

    return sanitized