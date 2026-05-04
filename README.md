# ResearchRabbit Backend

FastAPI backend for the ResearchRabbit Chrome extension. Deployed on DigitalOcean App Platform.

## What It Does

- Accepts page text and a research prompt
- Chunks the text and ranks chunks by semantic similarity using DO GenAI Inference Router embeddings
- Returns the top-k most relevant chunks

## Local Development

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\Activate.ps1 on Windows
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your DO_INFERENCE_TOKEN
uvicorn app.main:app --reload --port 8080
```

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `DO_INFERENCE_TOKEN` | Yes | — | DigitalOcean GenAI model access key |
| `EMBEDDING_MODEL` | No | `qwen3-embedding-0.6b` | Embedding model ID |
| `ALLOWED_ORIGINS` | No | `chrome-extension://*` | Comma-separated CORS origins |

## Deploy to DigitalOcean App Platform

The `.do/app.yaml` defines the app spec. Connect this repo to App Platform and it will auto-deploy on push to `main`.

## Tests

```bash
PYTHONPATH=. pytest
```

## API

`GET /health` → `{"status": "ok"}`

`POST /rank-chunks`

```json
{
  "prompt": "What dataset did the authors use?",
  "page_text": "Visible webpage text...",
  "top_k": 5
}
```
