# Physical AI & Humanoid Robotics Book with RAG Chatbot

This repository contains the implementation of a comprehensive educational platform consisting of a Docusaurus-based book for the Physical AI & Humanoid Robotics course with an integrated RAG (Retrieval-Augmented Generation) chatbot.

## Architecture Overview

The system implements a comprehensive educational platform consisting of:
1. **Docusaurus Frontend**: Static site generation with embedded chatbot widget
2. **FastAPI Backend**: RAG system with Qdrant vector storage and Neon Postgres logging
3. **Content Pipeline**: Markdown → embeddings → searchable index
4. **User Management**: Authentication, progress tracking, and assessment systems

## Tech Stack

- **Frontend**: Docusaurus v3+ with custom React components for chatbot integration
- **Backend**: FastAPI with Python 3.10+
- **Vector Database**: Qdrant Cloud for semantic search
- **Relational Database**: Neon Postgres for user data and logs
- **Authentication**: JWT-based with refresh tokens
- **Deployment**: GitHub Pages (frontend), Render/Fly.io (backend)

## Project Structure

```
├── book-frontend/          # Docusaurus-based book frontend
├── rag-backend/            # FastAPI-based RAG system backend
├── book-content/           # Book content in Markdown format
├── rag-system/             # RAG system components
│   ├── backend/            # Backend implementation
│   ├── frontend/           # Frontend components
│   └── deployment/         # Deployment configurations
└── specs/                  # Specification files
    └── 005-book-rag-system/ # This feature specifications
```

## Features

- 13-week comprehensive course on Physical AI & Humanoid Robotics
- Integrated RAG chatbot with selected-text mode functionality
- User progress tracking and assessment system
- Content management with automatic indexing
- Responsive design for different devices

## Getting Started

1. Clone the repository
2. Install dependencies for both frontend and backend
3. Set up the required services (Qdrant, Neon Postgres)
4. Run the development servers

## License

[License information to be added]
