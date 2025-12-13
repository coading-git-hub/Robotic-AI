# RAG Backend Setup

This document outlines how to set up the Python virtual environment for the RAG backend system.

## Prerequisites

- Python 3.10 or higher
- pip package manager

## Setup Instructions

### 1. Create Virtual Environment

```bash
python -m venv venv
```

### 2. Activate Virtual Environment

On Windows:
```bash
venv\Scripts\activate
```

On macOS/Linux:
```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Set up Environment Variables

Create a `.env` file in the rag-backend directory with the following variables:

```env
DATABASE_URL=postgresql+asyncpg://username:password@localhost/dbname
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=your_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
SECRET_KEY=your_secret_key_here
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
REFRESH_TOKEN_EXPIRE_DAYS=7
```

### 5. Run the Application

```bash
cd rag-backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The application will be available at `http://localhost:8000`.