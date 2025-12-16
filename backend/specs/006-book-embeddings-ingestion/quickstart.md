# Quickstart: Book Content Embeddings Ingestion

## Overview
This guide shows how to set up and run the book content embeddings ingestion pipeline that fetches content from deployed book URLs, processes it, generates Cohere embeddings, and stores them in Qdrant Cloud.

## Prerequisites
- Python 3.10+
- Git
- Access to Cohere API
- Access to Qdrant Cloud instance

## Setup

### 1. Clone the Repository
```bash
git clone <repository-url>
cd humanoid-robotics
```

### 2. Navigate to Backend Directory
```bash
cd backend
```

### 3. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Configure Environment Variables
Copy the example environment file:
```bash
cp .env.example .env
```

Edit `.env` and add your configuration:
```bash
# Qdrant Configuration
QDRANT_HOST="your-qdrant-host-url"
QDRANT_API_KEY="your-qdrant-api-key"
QDRANT_COLLECTION_NAME="book_embeddings"

# Cohere Configuration
COHERE_API_KEY="your-cohere-api-key"

# Book Source Configuration
BOOK_BASE_URL="https://your-book-site.com"
```

## Running the Ingestion Pipeline

### Basic Execution
```bash
python main.py
```

### With Specific Options
The script supports various command-line options:
```bash
# Process only specific number of URLs
python main.py --limit 10

# Process from a specific starting point
python main.py --offset 20

# Validate storage after processing
python main.py --validate
```

## Validation

### Check Qdrant Storage
Run the validation script to verify embeddings are properly stored:
```bash
python test_embeddings.py
```

This will:
- Test connection to Qdrant
- Verify sitemap access
- Check storage status
- Process a sample URL to validate the full pipeline

### Expected Output
Upon successful execution, you should see:
- All book URLs processed successfully
- Embeddings stored in Qdrant with complete metadata
- No duplicate vectors created on re-runs
- Complete metadata (URL, title, content, chunk index) preserved

## Troubleshooting

### Common Issues

1. **API Rate Limits**: If you encounter rate limit errors, the script has built-in retry logic with exponential backoff.

2. **Network Errors**: The script handles network failures gracefully and continues processing remaining URLs.

3. **Qdrant Connection**: Verify your QDRANT_HOST and QDRANT_API_KEY are correct in your environment.

### Logs
Check the console output for detailed progress and any errors during processing.

## Next Steps
- Integrate with your RAG system for retrieval
- Set up scheduled runs for content updates
- Monitor storage usage in Qdrant Cloud