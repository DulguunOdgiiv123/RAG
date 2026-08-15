"""Load documents from disk and split them into overlapping, citeable chunks.

Design choice: chunking is sentence-aware (never splits mid-sentence) rather
than a raw fixed-width slide, because ripping a sentence in half at the
embedding stage measurably hurts retrieval quality — the embedding ends up
representing a fragment instead of a coherent claim. This is the kind of
detail worth being able to explain in an interview: "why not just
`text[i:i+1000]`?"
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from ragqa.config import settings

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    chunk_index: int


def _read_txt_or_md(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def load_document(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return _read_pdf(path)
    if ext in {".txt", ".md"}:
        return _read_txt_or_md(path)
    raise ValueError(f"Unsupported file type: {ext}")


def split_sentences(text: str) -> list[str]:
    # Collapse whitespace first so the regex isn't fighting newlines.
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []
    return _SENTENCE_SPLIT_RE.split(normalized)


def chunk_text(
    text: str,
    source: str,
    chunk_size_chars: int = settings.chunk_size_chars,
    overlap_chars: int = settings.chunk_overlap_chars,
) -> list[Chunk]:
    """Greedily pack whole sentences into ~chunk_size_chars windows, with the
    last `overlap_chars` worth of sentences repeated at the start of the next
    chunk so retrieval doesn't lose context that straddled a chunk boundary.
    """
    sentences = split_sentences(text)
    chunks: list[Chunk] = []
    current: list[str] = []
    current_len = 0
    idx = 0

    def flush():
        nonlocal current, current_len, idx
        if not current:
            return
        chunk_str = " ".join(current)
        chunks.append(
            Chunk(id=f"{source}::chunk{idx}", text=chunk_str, source=source, chunk_index=idx)
        )
        idx += 1

    for sentence in sentences:
        sentence_len = len(sentence) + 1
        if current_len + sentence_len > chunk_size_chars and current:
            flush()
            # carry overlap: walk backward from the end of `current` until
            # we've accumulated ~overlap_chars, reuse those sentences
            overlap: list[str] = []
            overlap_len = 0
            for s in reversed(current):
                if overlap_len >= overlap_chars:
                    break
                overlap.insert(0, s)
                overlap_len += len(s) + 1
            current = overlap
            current_len = overlap_len
        current.append(sentence)
        current_len += sentence_len

    flush()
    return chunks


def ingest_directory(data_dir: str | Path) -> list[Chunk]:
    data_dir = Path(data_dir)
    all_chunks: list[Chunk] = []
    files = sorted(
        p for p in data_dir.rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not files:
        raise FileNotFoundError(
            f"No supported files ({SUPPORTED_EXTENSIONS}) found under {data_dir}"
        )
    for path in files:
        text = load_document(path)
        all_chunks.extend(chunk_text(text, source=path.name))
    return all_chunks
