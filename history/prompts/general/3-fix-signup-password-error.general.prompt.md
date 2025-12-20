---
id: 3
title: "Fix Signup Password Encoding Error by Migrating to Argon2"
stage: "general"
date: "2025-12-20"
surface: "agent"
model: "Claude Sonnet 4.5"
feature: "none"
branch: "1-auth-personalization"
user: "user"
command: "sp.implement"
labels: ["authentication", "password-hashing", "argon2", "signup", "fix", "security"]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
  - "backend/src/auth/services.py"
  - "backend/requirements.txt"
  - "backend/ARGON2_MIGRATION_SUMMARY.md"
  - "backend/test_argon2_fix.py"
  - "backend/test_final_signup.py"
tests:
  - "backend/test_argon2_fix.py"
  - "backend/test_final_signup.py"
---

# Fix Signup Password Encoding Error by Migrating to Argon2

## Summary
Fixed the signup error "Password encoding error: password cannot be longer than 72 bytes" by migrating from bcrypt to Argon2 password hashing algorithm. The original bcrypt implementation had backend compatibility issues even with 8-character ASCII passwords.

## Problem
Users encountered the error "Password encoding error: password cannot be longer than 72 bytes, truncate manually if necessary" during signup, even when using exactly 8 ASCII characters as required.

## Root Cause
The bcrypt library had backend compatibility issues on the user's system, causing the error even with properly formatted 8-character ASCII passwords.

## Solution
Migrated from bcrypt to Argon2 password hashing algorithm which:
- Is more secure than bcrypt
- Doesn't have the same backend compatibility issues
- Is the winner of the Password Hashing Competition
- Provides better resistance to GPU/rainbow table attacks

## Changes Made

### Backend Changes
1. **Password Hashing Implementation** (`backend/src/auth/services.py`):
   - Updated CryptContext to use argon2 instead of bcrypt
   - Updated configuration parameters for Argon2 (rounds=10, memory_cost=102400, parallelism=1)
   - Updated docstrings and comments to reflect Argon2 usage
   - Updated error messages to be more generic

2. **Dependencies** (`backend/requirements.txt`):
   - Added `argon2-cffi>=21.1.0` dependency

### Testing and Documentation
1. Created comprehensive test scripts to verify Argon2 functionality
2. Created migration summary documentation
3. Verified complete signup flow works correctly

## Verification
- Password hashing and verification work correctly with Argon2
- All 8-character ASCII passwords are properly handled
- Complete signup flow works with Neon database
- Authentication (signin) works correctly
- Wrong password authentication fails as expected
- No more bcrypt backend compatibility errors

## Security Improvements
- Argon2 is more secure than bcrypt
- Configurable memory hardness prevents GPU attacks
- Time-tested and recommended by security experts
- Maintains the same 8-character password requirement for consistency

## Compatibility
- All existing API endpoints remain unchanged
- Frontend authentication flow remains the same
- Database schema remains unchanged
- Password validation rules remain the same