# Member QnA API

AI-powered question-answering system for member messages. Ask questions in plain English and get instant answers.

**🚀 Live API:** https://message-qna.onrender.com/

**📚 Interactive Docs:** https://message-qna.onrender.com/docs

---

## What is this?

This API lets you ask natural language questions about member data and messages, and get AI-generated answers based on the stored information.

## How to Use

### Option 1: Try it in Your Browser

Visit the interactive documentation: **https://message-qna.onrender.com/docs**

1. Click on the **POST /ask** endpoint
2. Click **"Try it out"**
3. Enter your question in the JSON format
4. Click **"Execute"**

### Option 2: Use cURL (Command Line)

```bash
curl -X POST "https://message-qna.onrender.com/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "When is Layla planning her trip to London?"}'
```

### Option 3: Use Python

```python
import requests

response = requests.post(
    "https://message-qna.onrender.com/ask",
    json={"question": "When is Layla planning her trip to London?"}
)

print(response.json()["answer"])
```

### Option 4: Use JavaScript/Node.js

```javascript
fetch("https://message-qna.onrender.com/ask", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ 
    question: "When is Layla planning her trip to London?" 
  })
})
  .then(res => res.json())
  .then(data => console.log(data.answer));
```

---

## Example Questions to Try

- "When is Layla planning her trip to London?"
- "What restaurants has Michael mentioned?"
- "Who is interested in Italian food?"
- "What are Sarah's hobbies?"
- "Who has talked about traveling?"

## What You'll Get Back

The API returns a simple JSON response:

```json
{
  "answer": "Layla Kawaguchi is planning a trip to London in March 2024."
}
```

## Response Times

- **First request after deployment:** 2-4 seconds (the system needs to load)
- **After that:** 1-3 seconds per question

💡 **Tip:** The first request might be slower. Be patient!

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ask` | POST | Ask a question about member data |
| `/docs` | GET | Interactive API documentation (try this!) |
| `/redoc` | GET | Alternative documentation view |
| `/health` | GET | Check if the API is running |
| `/warmup` | POST | Pre-load the system (for faster first requests) |

---

## For Developers: Running Locally

If you want to run this on your own machine:

```bash
# Install dependencies
pip install -r requirements.txt

# Create your environment file
cp example.env .env

# Add your API keys to .env file
# OPENAI_API_KEY=your_key_here
# MESSAGES_API_KEY=your_key_here

# Run the application
python -m app.main
```

The API will start at http://localhost:8000

### Technical Documentation

For implementation details and optimization guides:
- [MEMORY_OPTIMIZATION_GUIDE.md](./MEMORY_OPTIMIZATION_GUIDE.md)
- [QUICK_REFERENCE.md](./QUICK_REFERENCE.md)
