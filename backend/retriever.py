from __future__ import annotations

import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

_EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _faiss_index_dir() -> Path:
    return _repo_root() / "backend" / "faiss_index"


@lru_cache(maxsize=1)
def _get_embeddings() -> HuggingFaceEmbeddings:
    # Must match the model used in `backend/ingest.py`.
    return HuggingFaceEmbeddings(model_name=_EMBEDDING_MODEL_NAME)


@lru_cache(maxsize=1)
def _load_vectorstore() -> FAISS:
    faiss_dir = _faiss_index_dir()
    if not faiss_dir.exists():
        raise FileNotFoundError(
            f"FAISS index not found at '{faiss_dir}'. Run 'backend/ingest.py' first."
        )

    # `save_local()` persists docstore + index; older LC versions require this flag.
    return FAISS.load_local(
        str(faiss_dir),
        embeddings=_get_embeddings(),
        allow_dangerous_deserialization=True,
    )


def _extract_source_filename(doc: Any) -> str:
    """
    Best-effort extraction of the original filename from a LangChain Document.
    """
    metadata = getattr(doc, "metadata", None) or {}

    source = metadata.get("source") or metadata.get("filename")
    if not source:
        return "unknown"

    # `PyPDFLoader` typically stores the input path in `metadata["source"]`.
    try:
        return Path(str(source)).name
    except Exception:
        return str(source)


def search(query: str, k: int = 3) -> List[Dict[str, Any]]:
    """
    Search the FAISS index and return top-k relevant chunks.

    Returns items like:
    {"source": "<filename>", "chunk": "<chunk text>", "score": <float>}.

    Note: the score comes from FAISS/LangChain and is distance-like (lower is better).
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    if k <= 0:
        raise ValueError("k must be a positive integer")

    vectorstore = _load_vectorstore()

    start = time.perf_counter()
    docs_with_scores = vectorstore.similarity_search_with_score(query, k=k)
    elapsed_ms = (time.perf_counter() - start) * 1000

    # Required: print response time in milliseconds.
    print(f"Retriever response time: {elapsed_ms:.2f} ms")

    results: List[Dict[str, Any]] = []
    for doc, score in docs_with_scores:
        results.append(
            {
                "source": _extract_source_filename(doc),
                "chunk": getattr(doc, "page_content", ""),
                "score": float(score),
            }
        )
    return results


def get_index_stats() -> int:
    """
    Return the total number of indexed text chunks in the saved FAISS index.
    """
    vectorstore = _load_vectorstore()

    # `index_to_docstore_id` maps vector positions to docstore keys.
    index_to_docstore_id = getattr(vectorstore, "index_to_docstore_id", None)
    if index_to_docstore_id is not None:
        try:
            return len(index_to_docstore_id)
        except Exception:
            pass

    # Fallback: docstore might store docs in `_dict` (InMemoryDocstore).
    docstore = getattr(vectorstore, "docstore", None)
    docstore_dict = getattr(docstore, "_dict", None)
    if docstore_dict is not None:
        try:
            return len(docstore_dict)
        except Exception:
            pass

    raise RuntimeError("Unable to determine FAISS index chunk count")


__all__ = ["search", "get_index_stats"]