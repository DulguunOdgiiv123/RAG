"""Central configuration. Override any of these with environment variables,
e.g. `RAGQA_GEN_MODEL=mistral:latest python -m ragqa.cli ask "..."`.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    # Ollama models — pull these first: `ollama pull nomic-embed-text && ollama pull llama3.1`
    embed_model: str = os.getenv("RAGQA_EMBED_MODEL", "nomic-embed-text")
    gen_model: str = os.getenv("RAGQA_GEN_MODEL", "llama3.1")

    # Ollama server (default is fine for local install)
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")

    # Chunking
    chunk_size_chars: int = int(os.getenv("RAGQA_CHUNK_SIZE", "1000"))
    chunk_overlap_chars: int = int(os.getenv("RAGQA_CHUNK_OVERLAP", "150"))

    # Retrieval
    top_k: int = int(os.getenv("RAGQA_TOP_K", "5"))

    # Storage
    persist_dir: str = os.getenv("RAGQA_PERSIST_DIR", "./chroma_store")
    collection_name: str = os.getenv("RAGQA_COLLECTION", "ragqa_docs")


settings = Settings()
