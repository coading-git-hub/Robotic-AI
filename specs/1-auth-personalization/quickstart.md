# Quickstart Guide: Authentication and Personalized Content System

## Prerequisites

- Node.js 18+ for frontend development
- Python 3.10+ for backend development
- Neon Postgres account with database created
- Better Auth account (for free tier usage)

## Environment Setup

### Backend Setup
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Frontend Setup
```bash
cd frontend
npm install
```

## Configuration

### Backend Configuration
Create `.env` file in the backend directory:
```env
DATABASE_URL="your-neon-postgres-connection-string"
BETTER_AUTH_SECRET="your-better-auth-secret"
BETTER_AUTH_URL="http://localhost:3000"
```

### Frontend Configuration
Create `.env.local` file in the frontend directory:
```env
NEXT_PUBLIC_API_URL="http://localhost:8000"
NEXT_PUBLIC_BETTER_AUTH_URL="http://localhost:3000/api/auth"
```

## Running the Application

### Backend (API Server)
```bash
cd backend
python -m uvicorn main:app --reload --port 8000
```

### Frontend (Development Server)
```bash
cd frontend
npm run dev
```

## Key Integration Points

### 1. User Registration with Background Collection
- Implement signup form with additional fields for software and hardware background
- Use Better Auth's custom fields functionality to store background information
- Validate background selections against allowed values

### 2. Profile Management
- Create user profile page accessible after authentication
- Allow users to update their background information
- Display current background selections clearly

### 3. Personalization Button Integration
- Add "Personalize" button to chapter pages
- Ensure button is only visible/functional when user is authenticated
- Implement click handler that calls personalization API

### 4. Content Adaptation
- Create content templates that can be parameterized based on user background
- Implement logic to fetch personalized content variants
- Update UI to indicate when content has been personalized

## API Endpoints

### Authentication Endpoints
- `POST /api/auth/signup` - Register new user with background info
- `POST /api/auth/signin` - Sign in existing user
- `GET /api/auth/profile` - Get current user profile
- `PUT /api/auth/profile` - Update user profile and background

### Personalization Endpoints
- `POST /api/personalization/personalize` - Trigger content personalization
- `GET /api/personalization/history` - Get personalization history

## Testing

### Backend Tests
```bash
cd backend
pytest tests/
```

### Frontend Tests
```bash
cd frontend
npm test
```

## Deployment

### Environment Variables for Production
Update environment variables for production deployment:
- Set proper database URL for production Neon instance
- Update API URLs to production endpoints
- Configure Better Auth for production domain

### Database Migration
Run database migrations to set up user profile extensions:
```bash
# After Better Auth creates initial tables, run the schema extension
psql $DATABASE_URL < database/extend_user_table.sql
```