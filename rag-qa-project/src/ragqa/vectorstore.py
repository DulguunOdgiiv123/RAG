"""Local, on-disk Chroma vector store. No server process required — it
persists to `settings.persist_dir` as SQLite + Parquet under the hood.
"""
from __future__ import annotations

import chromadb

from ragqa.config import settings
from ragqa.ingest import Chunk


class VectorStore:
    def __init__(self, persist_dir: str = settings.persist_dir, collection: str = settings.collection_name):
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=collection, metadata={"hnsw:space": "cosine"}
        )

    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        self._collection.add(
            ids=[c.id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[{"source": c.source, "chunk_index": c.chunk_index} for c in chunks],
        )

    def query(self, query_embedding: list[float], top_k: int = settings.top_k) -> list[dict]:
        result = self._collection.query(query_embeddings=[query_embedding], n_results=top_k)
        hits = []
        for doc, meta, dist, doc_id in zip(
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
            result["ids"][0],
        ):
            hits.append(
                {
                    "id": doc_id,
                    "text": doc,
                    "source": meta["source"],
                    "chunk_index": meta["chunk_index"],
                    # Chroma returns cosine *distance*; similarity = 1 - distance
                    "similarity": 1 - dist,
                }
            )
        return hits

    def count(self) -> int:
        return self._collection.count()
