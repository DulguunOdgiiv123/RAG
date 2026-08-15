"""Run with: uvicorn api.main:app --reload --port 8000
Then: curl -X POST localhost:8000/query -H "Content-Type: application/json" -d '{"question": "..."}'
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ragqa.pipeline import ask

app = FastAPI(title="ragqa", description="Local RAG Q&A over your corpus.")


class QueryRequest(BaseModel):
    question: str
    top_k: int | None = None


class SourceRef(BaseModel):
    index: int
    source: str
    similarity: float


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceRef]


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    try:
        result = ask(req.question, top_k=req.top_k)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return QueryResponse(**result)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
