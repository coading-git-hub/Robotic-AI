# Migration from bcrypt to Argon2 for Password Hashing

## Problem
The original implementation used bcrypt for password hashing but encountered backend compatibility issues causing the error:
"Password encoding error: password cannot be longer than 72 bytes, truncate manually if necessary (e.g. my_password[:72]). Please use only standard ASCII characters (letters, numbers, and basic symbols)."

This occurred even with 8-character ASCII passwords due to bcrypt backend initialization issues.

## Solution
Migrated from bcrypt to Argon2 password hashing algorithm which:

1. **More Secure**: Argon2 is the winner of the Password Hashing Competition and is considered more secure than bcrypt
2. **Better Compatibility**: Argon2 doesn't have the same backend compatibility issues as bcrypt
3. **Configurable**: Offers better tuning parameters for memory and CPU usage
4. **Modern Standard**: Recommended for new applications

## Changes Made

### Backend Changes
1. **Updated Password Context** (`src/auth/services.py`):
   - Changed from bcrypt to argon2 hashing scheme
   - Updated configuration parameters for Argon2
   - Added proper initialization for Argon2 backend

2. **Updated Hashing Method**:
   - Changed documentation and comments to reflect Argon2 usage
   - Updated error messages to be more generic
   - Maintained the same API for compatibility

3. **Updated Verification Method**:
   - Updated documentation to reflect Argon2 usage
   - Maintained the same API for compatibility

### Security Improvements
- Argon2 is resistant to side-channel attacks
- Configurable memory hardness to prevent GPU/rainbow table attacks
- Time-tested and recommended by security experts

## Configuration Details
- **Algorithm**: argon2 (specifically argon2i variant managed by passlib)
- **Rounds**: 10 iterations
- **Memory Cost**: 102400 KB (100MB)
- **Parallelism**: 1 thread

## Requirements
- Added `argon2-cffi` dependency for Argon2 support

## Compatibility
- All existing API endpoints remain unchanged
- Frontend authentication flow remains the same
- Database schema remains unchanged
- Password validation (8-character requirement) remains the same

## Testing
- Password hashing and verification work correctly with Argon2
- All 8-character ASCII passwords are properly handled
- No more bcrypt backend compatibility errors
- Complete user creation flow works properly