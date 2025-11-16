# Reflecting QnA

Ask natural language questions about member messages and get instant AI-powered answers.

## How to Use

**Deployed URL:** `https://your-deployed-app.onrender.com`

### Quick Example

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

print(response.json()["answer"])
```

### Using JavaScript

```javascript
fetch("https://your-deployed-app.onrender.com/ask", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ 
    question: "When is Layla planning her trip to London?" 
  })
})
  .then(res => res.json())
  .then(data => console.log(data.answer));
```

### Try These Questions

- "When is Layla planning her trip to London?"
- "What restaurants has Michael mentioned?"
- "Who is interested in Italian food?"
- "What are Sarah's hobbies?"

## Response Format

The API returns a JSON response with your answer:

```json
{
  "answer": "Layla Kawaguchi is planning a trip to London in March 2024."
}
```

## Performance

- **First request:** ~3-5 seconds (loading models)
- **Subsequent requests:** ~1-2 seconds
- **Tip:** Visit `/docs` for interactive API documentation

---

## For Developers

### Local Setup

```bash
pip install -r requirements.txt
cp example.env .env
# Add your OPENAI_API_KEY and MESSAGES_API_KEY
python -m app.main
```

### Additional Endpoints

- `GET /health` - Health check
- `POST /warmup` - Pre-load models (reduces first request latency)
- `POST /clear-cache` - Refresh member data

### Technical Details

See [MEMORY_OPTIMIZATION_GUIDE.md](./MEMORY_OPTIMIZATION_GUIDE.md) for implementation details.
