"""Turns retrieved chunks + a question into a grounded answer.

The system prompt does two things that matter for a portfolio piece:
1. Forces the model to only answer from provided context (reduces
   hallucination — you should be able to demonstrate this with a
   deliberately out-of-scope question in your eval set).
2. Forces inline citation tags like [1], [2] mapped back to source chunks,
   so answers are auditable — this is the difference between a RAG demo
   and a RAG system someone could actually trust.
"""
from __future__ import annotations

import ollama

from ragqa.config import settings

SYSTEM_PROMPT = """You are a precise Q&A assistant. You answer ONLY using the \
numbered context passages provided below. Rules:
- If the answer is not contained in the context, say "I don't have enough \
information in the provided documents to answer that" — do not guess.
- Every factual claim in your answer must end with a citation tag like [1] \
or [2] referring to the passage number it came from.
- Be concise. Do not repeat the passages verbatim; synthesize.
"""


def _format_context(hits: list[dict]) -> str:
    lines = []
    for i, hit in enumerate(hits, start=1):
        lines.append(f"[{i}] (source: {hit['source']})\n{hit['text']}")
    return "\n\n".join(lines)


def generate_answer(question: str, hits: list[dict], model: str = settings.gen_model) -> dict:
    context = _format_context(hits)
    user_prompt = f"Context passages:\n\n{context}\n\nQuestion: {question}\n\nAnswer:"

    client = ollama.Client(host=settings.ollama_host)
    response = client.chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    return {
        "answer": response["message"]["content"],
        "sources": [
            {"index": i + 1, "source": h["source"], "similarity": round(h["similarity"], 3)}
            for i, h in enumerate(hits)
        ],
    }
