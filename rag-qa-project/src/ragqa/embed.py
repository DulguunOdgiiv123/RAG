"""Thin wrapper over Ollama's embedding endpoint, batched to keep ingestion
of a few hundred chunks from taking forever on CPU.
"""
from __future__ import annotations

import ollama

from ragqa.config import settings


def embed_texts(texts: list[str], model: str = settings.embed_model) -> list[list[float]]:
    """Embed a list of strings. Ollama's Python client doesn't batch natively
    (one HTTP call per text as of writing), so we just loop — fine for
    portfolio-scale corpora (hundreds to low thousands of chunks). If you
    outgrow this, that's a legitimate reason to swap in a batched local
    model via sentence-transformers instead — worth noting as a documented
    tradeoff rather than silently eating the slowdown.
    """
    client = ollama.Client(host=settings.ollama_host)
    vectors: list[list[float]] = []
    for text in texts:
        response = client.embeddings(model=model, prompt=text)
        vectors.append(response["embedding"])
    return vectors


def embed_query(query: str, model: str = settings.embed_model) -> list[float]:
    return embed_texts([query], model=model)[0]
