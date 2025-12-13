# Neon Postgres Database Setup

This document outlines how to set up the Neon Postgres database for the RAG system.

## Cloud Setup

### 1. Create Neon Account
- Go to https://neon.tech/
- Sign up for a free account or log in if you already have one
- Create a new project with appropriate region and settings

### 2. Database Configuration
- Choose PostgreSQL version 14 or higher
- Select an appropriate region for low latency
- Set up connection pooling if needed for high concurrency

### 3. Environment Configuration
Add the following to your `.env` file:

```env
DATABASE_URL=postgresql://username:password@ep-xxx.us-east-1.aws.neon.tech/dbname?sslmode=require
NEON_PROJECT_ID=your_project_id
NEON_API_KEY=your_api_key
```

## Local Development Setup (Alternative)

For local development, you can use a local PostgreSQL instance:

1. Install PostgreSQL (version 14+)
2. Create a database:
```sql
CREATE DATABASE physical_ai_rag;
CREATE USER rag_user WITH PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE physical_ai_rag TO rag_user;
```

3. Use this configuration in your `.env`:
```env
DATABASE_URL=postgresql+asyncpg://rag_user:secure_password@localhost:5432/physical_ai_rag
```

## Schema Setup

The database schema is defined in `DATABASE_SCHEMA.sql` and includes:

- **users**: User account information and profiles
- **progress_records**: Tracking of user progress through course content
- **chat_sessions**: Conversation sessions with the RAG chatbot
- **queries**: Log of all queries made to the RAG system
- **system_configs**: Configuration parameters for the system
- **notifications**: System notifications for users

## Running Migrations

The application uses Alembic for database migrations:

1. Install alembic: `pip install alembic`
2. Initialize: `alembic init alembic`
3. Generate migration: `alembic revision --autogenerate -m "Initial schema"`
4. Apply migration: `alembic upgrade head`

## Connection Pooling

The application will use SQLAlchemy's built-in connection pooling with the following configuration:
- Pool size: 20 connections
- Max overflow: 30 connections
- Pool timeout: 30 seconds
- Pool recycling: 3600 seconds (1 hour)