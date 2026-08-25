"""
StudySage — RAG Chat Service
Retrieval-Augmented Generation using:
  - sentence-transformers for embeddings
  - FAISS for vector search
  - OpenAI-compatible Anthropic API for generation
Falls back to context-window chat when vector store is empty.
"""
from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ── Optional: sentence-transformers + FAISS ────────────────────────────────────
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    _embed_model = SentenceTransformer(
        os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
    )
    _EMBED = True
    logger.info("SentenceTransformer loaded (all-MiniLM-L6-v2)")
except Exception as e:
    _EMBED = False
    _embed_model = None
    logger.warning("sentence-transformers not available: %s", e)

try:
    import faiss  # type: ignore
    _FAISS = True
    logger.info("FAISS available")
except ImportError:
    _FAISS = False
    logger.warning("FAISS not installed — RAG retrieval disabled, using full context")


# ── In-memory vector store (per-process) ──────────────────────────────────────
# Structure: { user_id: { "index": faiss.IndexFlatIP, "chunks": [str] } }
_stores: dict[str, dict[str, Any]] = {}


def _chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
    """Split text into overlapping word chunks."""
    words = text.split()
    chunks: list[str] = []
    step = chunk_size - overlap
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def index_document(user_id: str, text: str) -> int:
    """
    Embed and index document text for a user.
    Returns number of chunks indexed.
    """
    if not (_EMBED and _FAISS):
        logger.info("RAG indexing skipped (missing deps) for user %s", user_id)
        return 0

    chunks = _chunk_text(text)
    if not chunks:
        return 0

    embeddings = _embed_model.encode(chunks, normalize_embeddings=True)
    dim = embeddings.shape[1]

    if user_id not in _stores:
        index = faiss.IndexFlatIP(dim)  # Inner product = cosine similarity on normalised vecs
        _stores[user_id] = {"index": index, "chunks": []}

    store = _stores[user_id]
    store["index"].add(np.array(embeddings, dtype=np.float32))
    store["chunks"].extend(chunks)
    logger.info("Indexed %d chunks for user %s (total=%d)", len(chunks), user_id, len(store["chunks"]))
    return len(chunks)


def retrieve_context(user_id: str, query: str, top_k: int = 4) -> str:
    """Retrieve top-k relevant chunks for a query."""
    if not (_EMBED and _FAISS) or user_id not in _stores:
        return ""

    store = _stores[user_id]
    if store["index"].ntotal == 0:
        return ""

    q_emb = _embed_model.encode([query], normalize_embeddings=True)
    scores, indices = store["index"].search(np.array(q_emb, dtype=np.float32), top_k)

    chunks = store["chunks"]
    results = [chunks[i] for i in indices[0] if 0 <= i < len(chunks)]
    return "\n\n---\n\n".join(results)


def clear_index(user_id: str) -> None:
    """Clear vector store for a user (e.g. on new upload)."""
    _stores.pop(user_id, None)


def build_rag_system_prompt(
    context: str,
    topics: list[str],
    language: str,
    difficulty: str,
) -> str:
    ctx_section = f"\n\n## Relevant study material\n{context}\n" if context else ""
    topic_section = f"\nCovered topics: {', '.join(topics)}" if topics else ""
    return (
        f"You are Syllabus Slayer, an expert AI study tutor.{ctx_section}{topic_section}\n\n"
        f"Instructions:\n"
        f"- Answer clearly and helpfully in **{language}**.\n"
        f"- Target **{difficulty}** level understanding.\n"
        f"- Be concise — prefer bullet points for lists.\n"
        f"- If the answer is in the study material above, cite it.\n"
        f"- If you don't know, say so rather than guessing.\n"
        f"- Format math/code in markdown.\n"
    )
