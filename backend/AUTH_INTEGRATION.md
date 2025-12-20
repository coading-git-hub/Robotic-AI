# Better-Auth Integration with Neon Database

## Overview
This project implements a custom authentication system that mimics better-auth's API structure while using Neon Postgres database for storing user data. The implementation provides compatibility with better-auth's frontend client while using a custom backend solution.

## Key Features
- **Neon Database Integration**: Properly configured SQLAlchemy models that work with Neon Postgres
- **Better-Auth Compatible API**: Endpoints that match better-auth's response format
- **Secure Password Handling**: Proper bcrypt hashing with 8-character password requirement
- **JWT Authentication**: Secure token-based authentication system

## API Endpoints
- `POST /api/auth/signup` - Standard signup endpoint
- `POST /api/auth/sign-up/email` - Better-auth compatible endpoint
- `POST /api/auth/signin` - Login endpoint
- `GET /api/auth/profile` - Get user profile
- `PUT /api/auth/profile` - Update user profile

## Database Configuration
The system uses the following environment variables for database configuration:
- `NEON_DATABASE_URL`: Primary database URL for Neon Postgres
- `DATABASE_URL`: Fallback database URL
- Falls back to SQLite for development if no database URL is provided

## Password Requirements
- Exactly 8 characters required
- ASCII characters only (to avoid bcrypt 72-byte limit issues)
- Properly hashed using bcrypt with 12 rounds

## Frontend Integration
The frontend uses `BetterAuthProvider` to manage authentication state and provides:
- `signup()` - Register new users
- `signin()` - Login existing users
- `signout()` - Logout users
- `getProfile()` - Get current user profile
- `updateProfile()` - Update user information

## Database Models
- `User` model with email, name, password hash, and background information
- `PersonalizationSession` model for tracking personalization sessions

## Error Handling
- Proper database transaction rollback on errors
- Comprehensive error messages for debugging
- Validation for all input fields
- Connection pooling for Neon database

## Testing
Use the test script `test_signup_neon.py` to verify the signup functionality works with Neon database.

## Troubleshooting
1. **Signup fails**: Check that password is exactly 8 characters
2. **Database connection issues**: Verify NEON_DATABASE_URL in .env file
3. **Tables not created**: Ensure create_tables() is called on startup
4. **Server not responding**: Make sure the server is running on the correct port