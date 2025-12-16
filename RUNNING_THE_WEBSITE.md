# Running the Physical AI Humanoid Robotics Website with Chatbot

This guide explains how to run both the backend RAG agent and the frontend website with full chatbot functionality.

## Prerequisites

- Node.js (v18 or higher)
- Python (v3.8 or higher)
- pip (Python package manager)

## Backend Setup (RAG Agent)

### 1. Navigate to the backend directory
```bash
cd backend
```

### 2. Install Python dependencies
```bash
pip install -r requirements.txt
```

The requirements.txt file contains:
- cohere>=4.5.0
- qdrant-client>=1.7.0
- python-dotenv>=1.0.0
- requests>=2.31.0

If you need additional dependencies for the FastAPI server:
```bash
pip install fastapi uvicorn
```

### 3. Set up environment variables
Create a `.env` file in the backend directory with the following content:

```env
# Cohere Configuration
COHERE_API_KEY="your-cohere-api-key-here"
QDRANT_URL="your-qdrant-cloud-url-here"
QDRANT_API_KEY="your-qdrant-api-key-here"

# Additional Configuration (these are defaults, adjust as needed)
COHERE_MODEL=command-r-plus
EMBEDDING_MODEL=embed-english-v3.0
CONTEXT_WINDOW_LIMIT=120000
SELECTED_TEXT_PRIORITY=0.8
FALLBACK_MESSAGE=I cannot answer based on the provided context.
MAX_QUERY_LENGTH=2000
MAX_SELECTED_TEXT_LENGTH=5000
TOP_K_RESULTS=5
SIMILARITY_THRESHOLD=0.3
RETRIEVAL_TIMEOUT=10
```

### 4. Run the backend server
```bash
python rag_agent_api.py
```

Or using uvicorn directly:
```bash
uvicorn rag_agent_api:app --host 0.0.0.0 --port 8000
```

The backend should now be running on `http://localhost:8000`

You can verify the backend is working by visiting:
- Health check: `http://localhost:8000/api/health`
- API documentation: `http://localhost:8000/docs`

## Frontend Setup (Docusaurus Website)

### 1. Navigate to the frontend directory
```bash
cd book-frontend
```

### 2. Install dependencies
```bash
npm install
```

### 3. Set up environment variables
Create a `.env` file in the book-frontend directory with the following content:

```env
REACT_APP_BACKEND_URL=http://localhost:8000
BACKEND_URL=http://localhost:8000
```

### 4. Run the frontend development server
```bash
npm start
```

The website will be available at `http://localhost:3000`

## Step-by-Step Process to Run Both Services

### Option 1: Sequential Setup (Recommended for first time)

1. **Open a terminal/command prompt** and navigate to the project root:
   ```bash
   cd C:\Users\FATTANI COMPUTERS\Desktop\Physical-AI\humanoid-robotics
   ```

2. **Start the backend server** (first terminal window):
   ```bash
   cd backend
   python rag_agent_api.py
   ```

   Keep this terminal window open - the backend needs to stay running.

3. **Open a new terminal/command prompt** (second window), navigate to the project root again:
   ```bash
   cd C:\Users\FATTANI COMPUTERS\Desktop\Physical-AI\humanoid-robotics
   ```

4. **Start the frontend server** (second terminal window):
   ```bash
   cd book-frontend
   npm start
   ```

5. **Open your browser** and go to `http://localhost:3000`

### Option 2: Using Concurrent Tasks (Advanced)

If you have `concurrently` installed, you can run both services from the project root:

```bash
npm install -g concurrently
concurrently "cd backend && python rag_agent_api.py" "cd book-frontend && npm start"
```

## Testing the Complete Integration

1. **Verify the backend is running**: Visit `http://localhost:8000/api/health` - you should get a JSON response showing the service status

2. **Access the website**: Visit `http://localhost:3000` in your browser

3. **Check chatbot functionality**:
   - You should see a floating chat icon in the bottom right corner
   - The chat header should show the backend connection status (Online/Offline/Checking)
   - Try selecting text on any page - the chat should open automatically
   - Type questions in the chat to verify the agent responds properly
   - Check that selected text appears in the chat context

4. **Verify both services are communicating**:
   - Open browser developer tools (F12)
   - Go to the Network tab
   - Ask a question in the chat
   - You should see requests being made to `http://localhost:8000/api/agent/query`

## Alternative: Production Build

If you want to build and serve a production version:

### 1. Build the frontend
```bash
cd book-frontend
npm run build
```

### 2. Serve the built site
```bash
npm run serve
```

Note: For production builds, you'll need to ensure the backend URL is properly configured in your deployment environment.

## Troubleshooting

### Backend not connecting:
- Verify the backend server is running on `http://localhost:8000`
- Check the backend logs for any errors
- Ensure your environment variables are properly set with valid API keys
- Verify that port 8000 is not being used by another application

### Frontend not connecting to backend:
- Check that the `REACT_APP_BACKEND_URL` environment variable is set correctly
- Verify that both services are running simultaneously
- Check browser console for CORS or network errors

### Chat not opening on text selection:
- Ensure both frontend and backend are running
- Check browser console for any JavaScript errors
- Verify that you're selecting enough text (empty selections won't trigger the chat)

### Text selection not working:
- Try selecting text on different parts of the page
- Check browser console for errors
- Ensure JavaScript is enabled in your browser
- Verify that the TextSelectionHandler is properly integrated

### Agent not responding:
- Verify your Cohere and Qdrant API keys are valid
- Check that your embeddings have been properly ingested into Qdrant
- Look at the backend logs for specific error messages

## Required Services

Before the chatbot can fully function, you need:
1. **Qdrant Vector Database**: Make sure your embeddings have been ingested
2. **Cohere API Access**: Valid API key for embedding and generation
3. **Backend RAG Agent**: Running and connected to both Qdrant and Cohere
4. **Frontend Website**: Running and connected to the backend

## Data Preparation (If needed)

If you haven't ingested your book data yet, you may need to run the ingestion script:

```bash
cd backend
python ingest_all_data.py
```

This will process your book content and store the embeddings in Qdrant for the RAG system to use.