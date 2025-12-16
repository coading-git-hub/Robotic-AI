# Quick Start Guide: Running the Physical AI Website with Chatbot

## Overview
This project consists of two main components:
1. **Backend**: RAG Agent API running on port 8000
2. **Frontend**: Docusaurus website with chatbot running on port 3000

## Prerequisites
- Python 3.8+ with pip
- Node.js 18+ with npm
- Valid API keys for Cohere and Qdrant

## Step-by-Step Setup

### 1. Backend Setup (Terminal 1)
```bash
# Navigate to backend directory
cd backend

# Install Python dependencies
pip install -r requirements.txt
pip install fastapi uvicorn  # Additional dependencies for the server

# Create .env file with your API keys
# (Use the example values from RUNNING_THE_WEBSITE.md)

# Start the backend server
python rag_agent_api.py
```

### 2. Frontend Setup (Terminal 2)
```bash
# In a new terminal, navigate to frontend directory
cd book-frontend

# Install npm dependencies
npm install

# Create .env file with backend URL
# REACT_APP_BACKEND_URL=http://localhost:8000

# Start the frontend development server
npm start
```

### 3. Access the Website
Open your browser and go to: http://localhost:3000

## Expected Behavior
- You should see a floating chat icon in the bottom right
- When you select text on the page, the chat should open automatically
- You can ask questions and get responses from the RAG agent
- The chat header shows connection status (Online/Offline/Checking)

## Troubleshooting Quick Fixes
- Ensure both terminals are running their respective servers
- Check that ports 8000 (backend) and 3000 (frontend) are available
- Verify your API keys are valid in the backend .env file
- Check browser console for any JavaScript errors