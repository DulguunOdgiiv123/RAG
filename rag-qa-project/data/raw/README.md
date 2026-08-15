# ragqa — local RAG Q&A over your own documents

Fully local: embeddings + generation both run through Ollama, retrieval
runs through an on-disk Chroma index. No API keys, no per-query cost.

## Setup

```bash
# 1. Pull the models this project uses (one-time)
ollama pull nomic-embed-text
ollama pull llama3.1        # or swap for whatever chat model you already have

# 2. Install the project (editable install so `ragqa` is importable)
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Run it

```bash
# Drop .txt / .md / .pdf files into data/raw/, then:
python -m ragqa.cli build ./data/raw

# Ask questions against the index:
python -m ragqa.cli ask "What are the main complaints in these documents?"

# Or serve it as an API:
uvicorn api.main:app --reload --port 8000
curl -X POST localhost:8000/query -H "Content-Type: application/json" \
     -d '{"question": "What are the main complaints?"}'
```

## Tests

```bash
pytest tests/ -v
```
Chunking logic is tested without touching Ollama/Chroma, so `pytest` runs
fast and offline. Embedding/retrieval/generation are integration-tested
manually via the CLI against your running Ollama instance.

## Architecture

```
data/raw/*.{txt,md,pdf}
        │
        ▼
   ingest.py    — load + sentence-aware chunking (never splits a sentence)
        │
        ▼
   embed.py     — Ollama embeddings (nomic-embed-text)
        │
        ▼
  vectorstore.py — Chroma, persisted to ./chroma_store
        │
        ▼ (at query time)
  embed.py (query) → vectorstore.query() → top-k chunks
        │
        ▼
  generate.py   — Ollama chat model, forced to cite [1][2][3] per claim
        │
        ▼
   answer + source list
```

## Known limitations / next steps (in priority order)

1. **No retrieval evaluation yet.** Build a 20-30 question gold-answer set
   in `eval/`, measure recall@k (did the right chunk get retrieved at all)
   separately from answer quality (did the model use it correctly). This
   is the single highest-value thing to add — it's the difference between
   "I built a RAG demo" and "I understand RAG's actual failure modes" in
   an interview.
2. **Retrieval is vector-only.** Pure embedding similarity misses exact
   keyword/entity matches (e.g. a specific order ID or product code).
   Add BM25 (`rank_bm25` package) alongside vector search and combine
   scores (reciprocal rank fusion is the standard, simple approach) —
   this is "hybrid retrieval" and it's a very commonly asked-about
   technique.
3. **Embedding is unbatched.** `embed.py` loops one HTTP call per chunk.
   Fine at portfolio scale; document it as a known tradeoff, don't fix it
   unless you actually hit a wall — premature optimization here is wasted
   effort.
4. **No reranking step.** After vector retrieval pulls top-20, a cheap
   cross-encoder reranker narrows to top-5 more accurately than raw
   cosine similarity alone. Optional stretch once the basics work.
5. **Dockerize** the API (`Dockerfile` + `docker-compose.yml` alongside an
   `ollama` service) so the whole thing is a `docker compose up` — this is
   what makes it look like an actual deployable system rather than a
   personal script.
