# SHL Conversational Assessment Recommender

A conversational AI agent that helps hiring managers and recruiters find the right SHL Individual Test Solutions through natural dialogue.

## Architecture

```
User ──POST /chat──► FastAPI ──► Agent (OpenRouter)
                                    │
                                    ├── FAISS Semantic Search (MiniLM-L6-v2)
                                    │       └── SHL Catalog (377 items)
                                    │
                                    └── Scope Guard + Response Validator
```

## Quick Start

### 1. Install Dependencies

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Set Up Environment

```bash
cp .env.example .env
# Edit .env and add your OpenRouter API key
```

### 3. Run the Server

```bash
python main.py
# or
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Test It

```bash
# Health check
curl http://localhost:8000/health

# Chat
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "I need assessments for a Java developer"}
    ]
  }'
```

## API Reference

### `GET /health`

Returns `{"status": "ok"}` with HTTP 200.

### `POST /chat`

**Request:**
```json
{
  "messages": [
    {"role": "user", "content": "Hiring a Java developer who works with stakeholders"},
    {"role": "assistant", "content": "Sure. What is seniority level?"},
    {"role": "user", "content": "Mid-level, around 4 years"}
  ]
}
```

**Response:**
```json
{
  "reply": "Here are 5 assessments that fit a mid-level Java dev...",
  "recommendations": [
    {"name": "Java 8 (New)", "url": "https://www.shl.com/...", "test_type": "K"},
    {"name": "OPQ32r", "url": "https://www.shl.com/...", "test_type": "P"}
  ],
  "end_of_conversation": false
}
```

## Auto-Update Scraper

```bash
# Refresh the JSON catalog
python scraper.py

# Full deep scrape (JSON + website crawl)
python scraper.py --full

# Scheduled refresh every 24 hours
python scraper.py --schedule 24
```

## Project Structure

```
SHL Agent/
├── main.py                    # FastAPI endpoints
├── agent.py                   # OpenRouter-powered conversational agent
├── retriever.py               # FAISS semantic search engine
├── catalog_loader.py          # Data ingestion and merging
├── models.py                  # Pydantic request/response schemas
├── scraper.py                 # Auto-update catalog scraper
├── shl_product_catalog.json   # Official SHL catalog (primary data)
├── shl_catalog.csv            # Enriched scraped data
├── requirements.txt           # Python dependencies
├── Procfile                   # Deployment config
├── .env.example               # Environment template
└── approach_document.md       # Design approach writeup
```

## Deployment

### Render

1. Push to GitHub
2. Connect repo on [Render](https://render.com)
3. Set environment variable: `OLLAMA_BASE_URL` (or switch back to OpenRouter)
4. Build command: `pip install -r requirements.txt`
5. Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`

### Railway

1. Push to GitHub
2. Connect repo on [Railway](https://railway.app)
3. Set environment variable: `GEMINI_API_KEY`
4. Railway auto-detects the Procfile
