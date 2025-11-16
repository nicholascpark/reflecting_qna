# Reflecting QnA

A RAG-based question-answering service that uses LangGraph, FAISS vector search, and OpenAI to answer natural language questions about member messages.

## Overview

FastAPI service that performs semantic search over member data using:
- **FAISS** for vector storage and similarity search
- **OpenAI embeddings** (text-embedding-3-small) for semantic retrieval
- **LangGraph** for workflow orchestration
- **gpt-4o-mini** for answer generation

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment
cp example.env .env
# Add your OPENAI_API_KEY

# Run the service
python -m app.main
```

Visit `http://localhost:8000/docs` for interactive API documentation.

## API Endpoints

- `POST /ask` - Ask questions about member data
- `POST /warmup` - Pre-load FAISS index (reduces cold start)
- `POST /clear-cache` - Clear cache to refresh data
- `GET /health` - Health check

## Reflections

### Bonus 1: I wish I had created an ontology / knowledge graph based on the entity relationships to make this more accurate

A knowledge graph would capture explicit relationships between members (friends, family, colleagues) and entities (restaurants, activities, preferences), enabling more precise multi-hop reasoning beyond semantic similarity alone.
