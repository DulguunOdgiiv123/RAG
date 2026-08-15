from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.progress import track

from ragqa.embed import embed_query, embed_texts
from ragqa.generate import generate_answer
from ragqa.ingest import ingest_directory
from ragqa.vectorstore import VectorStore
from ragqa.config import settings

console = Console()


def build_index(data_dir: str | Path, batch_size: int = 16) -> VectorStore:
    """Ingest, chunk, embed, and persist a directory of documents."""
    chunks = ingest_directory(data_dir)
    console.print(f"[bold]{len(chunks)}[/bold] chunks produced from source documents.")

    store = VectorStore()
    for i in track(range(0, len(chunks), batch_size), description="Embedding + indexing"):
        batch = chunks[i : i + batch_size]
        vectors = embed_texts([c.text for c in batch])
        store.add_chunks(batch, vectors)

    console.print(f"Index built. Collection now holds [bold]{store.count()}[/bold] chunks.")
    return store


def ask(question: str, top_k: int | None = None) -> dict:
    """Query the existing index and return a grounded, cited answer."""
    store = VectorStore()
    if store.count() == 0:
        raise RuntimeError("Vector store is empty — run build_index() first.")

    q_vector = embed_query(question)
    hits = store.query(q_vector, top_k=top_k or settings.top_k)
    result = generate_answer(question, hits)
    result["question"] = question
    return result
