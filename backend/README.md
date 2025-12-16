# RAG Retrieval Validation Pipeline

This project implements a RAG (Retrieval Augmented Generation) retrieval validation pipeline that validates the retrieval process by processing user queries through Cohere embeddings and retrieving relevant content from Qdrant.

## Prerequisites

- Python 3.8 or higher
- pip package manager

## Virtual Environment Setup

It's recommended to use a virtual environment to manage dependencies for this project. Follow these steps:

1. Create a virtual environment:
   ```bash
   python -m venv venv
   ```

2. Activate the virtual environment:
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Deactivate the virtual environment when done:
   ```bash
   deactivate
   ```

## Configuration

Create a `.env` file based on the `.env.example` to configure the application:

```bash
cp .env.example .env
```

Then fill in your actual API keys and configuration values:

- `COHERE_API_KEY`: Your Cohere API key
- `QDRANT_URL`: Your Qdrant cloud URL
- `QDRANT_API_KEY`: Your Qdrant API key
- `TOP_K_RESULTS`: Number of results to retrieve (default: 5)
- `SIMILARITY_THRESHOLD`: Minimum similarity threshold (default: 0.3)
- `RETRIEVAL_TIMEOUT`: Timeout for retrieval operations (default: 10)

## Usage

To run the retrieval validation pipeline:

```bash
python retrieval.py
```

## Project Structure

- `retrieval.py`: Main script for the RAG retrieval validation pipeline
- `requirements.txt`: Python dependencies
- `.env.example`: Example environment configuration file
- `ingest_all_data.py`: Script to ingest book data into Qdrant
- `main.py`: Main application with ingestion functions