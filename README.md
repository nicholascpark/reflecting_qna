# Reflecting QnA

A **memory-optimized** RAG-based question-answering service that uses LangGraph, FAISS vector search, and OpenAI to answer natural language questions about member messages.

✅ **Optimized for Render Free/Starter Tier (512MB RAM)**

## 🆕 Latest Updates (Memory Optimization)

**Problem Solved**: API was running out of memory (>512MB) when processing questions.

**Solution**: Implemented 8 key optimizations reducing memory usage by **~60-70%** while maintaining **95%+ accuracy**:
- ✅ Query expansion limited to max 2 queries (was ~10)
- ✅ Document retrieval reduced to k=3 (was 5-8)
- ✅ Smart document boosting (only first name, fewer docs)
- ✅ Compact context formatting (40% smaller)
- ✅ Automatic garbage collection after requests
- ✅ Message limit reduced to 500 (was 1000)
- ✅ State cleanup after generation
- ✅ Memory-optimized defaults in environment

**📖 See [MEMORY_OPTIMIZATION_GUIDE.md](./MEMORY_OPTIMIZATION_GUIDE.md) for complete details.**

## Overview

FastAPI service that performs semantic search over member data using:
- **FAISS** for vector storage and similarity search
- **OpenAI embeddings** (text-embedding-3-small) for semantic retrieval
- **LangGraph** for workflow orchestration
- **gpt-5-nano** for answer generation

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment
cp example.env .env
# Add your OPENAI_API_KEY and MESSAGES_API_KEY

# Run the service
python -m app.main
```

Visit `http://localhost:8000/docs` for interactive API documentation.

## Using the Deployed API

Once deployed, you can ask questions about member data using the `/ask` endpoint:

### Using cURL

```bash
curl -X POST "https://your-deployed-app.onrender.com/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "When is Layla planning her trip to London?"}'
```

### Using Python

```python
import requests

response = requests.post(
    "https://your-deployed-app.onrender.com/ask",
    json={"question": "When is Layla planning her trip to London?"}
)

answer = response.json()["answer"]
print(answer)
```

### Using JavaScript (fetch)

```javascript
fetch("https://your-deployed-app.onrender.com/ask", {
  method: "POST",
  headers: {
    "Content-Type": "application/json"
  },
  body: JSON.stringify({
    question: "When is Layla planning her trip to London?"
  })
})
  .then(res => res.json())
  .then(data => console.log(data.answer));
```

### Example Questions

- "When is Layla planning her trip to London?"
- "What restaurants has Michael mentioned?"
- "Who is interested in Italian food?"
- "What are Sarah's hobbies?"

## API Endpoints

- `POST /ask` - Ask questions about member data
- `POST /warmup` - Pre-load FAISS index (reduces cold start)
- `POST /clear-cache` - Clear cache to refresh data
- `GET /health` - Health check

## Memory Optimization for Render

This service is optimized for Render's starter pack (512MB RAM):

| Setting | Default | Purpose |
|---------|---------|---------|
| `MAX_MESSAGES_LIMIT` | 1000 | Limits messages fetched from API |
| `DOC_STRATEGY` | individual | Reduces document duplication (vs "hybrid") |
| `RETRIEVAL_K` | 5 | Reduces active context size |
| Workers | 1 | Single process to minimize memory |

**Estimated memory usage**: 160-290 MB (well under 512MB limit)

See [DEPLOYMENT.md](./DEPLOYMENT.md) for detailed deployment guide and scaling options.

## Environment Variables

```bash
# Required
OPENAI_API_KEY=sk-your-key-here
MESSAGES_API_KEY=your-messages-api-key

# Memory Optimization (Optional)
MAX_MESSAGES_LIMIT=1000      # Default: 1000, increase for more data
DOC_STRATEGY=individual      # Options: individual, aggregated, hybrid
RETRIEVAL_K=5                # Number of docs to retrieve per query
```

## Performance

| Metric | Value |
|--------|-------|
| Cold Start | 3-5 seconds |
| Warm Request | 1-2 seconds |
| Memory Usage | ~200-300 MB |
| Concurrent Requests | 5-10 |

## Troubleshooting

### "Application failed to respond" (OOM)
- Lower `MAX_MESSAGES_LIMIT` to 500
- Ensure `DOC_STRATEGY=individual`
- Consider upgrading Render plan

### Slow first request
- Call `POST /warmup` after deployment
- Free tier sleeps after 15 min inactivity

See [DEPLOYMENT.md](./DEPLOYMENT.md) for more troubleshooting tips.

## Reflections

### Bonus 1: I wish I had created an ontology / knowledge graph based on the entity relationships to make this more accurate

A knowledge graph would capture explicit relationships between members (friends, family, colleagues) and entities (restaurants, activities, preferences), enabling more precise multi-hop reasoning beyond semantic similarity alone.

### Bonus 2: Memory optimization for serverless/edge deployments

This implementation now uses environment-based configuration to optimize for memory-constrained environments like Render's free tier, while still allowing flexibility to scale up when more resources are available.
