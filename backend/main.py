from __future__ import annotations

import time
from typing import Any, Dict

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.agents import run_query
from backend.retriever import get_index_stats
from backend.schemas import DocumentInfo, QueryRequest, QueryResponse

load_dotenv()

app = FastAPI(title="AI Finance Assistant API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _safe_index_stats() -> int:
    try:
        return int(get_index_stats())
    except FileNotFoundError:
        return 0


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "chunks_indexed": _safe_index_stats()}


@app.post("/query", response_model=QueryResponse)
def query(payload: QueryRequest) -> QueryResponse:
    start = time.perf_counter()
    final_state = run_query(payload.question)
    _elapsed_ms = (time.perf_counter() - start) * 1000

    return QueryResponse(
        answer=str(final_state.get("answer", "") or ""),
        sources=list(final_state.get("sources", []) or []),
        confidence=float(final_state.get("confidence", 0.0) or 0.0),
        tokens_used=int(final_state.get("tokens_used", 0) or 0),
    )


@app.get("/docs-info", response_model=DocumentInfo)
def docs_info() -> DocumentInfo:
    chunks = _safe_index_stats()
    return DocumentInfo(
        filename="faiss_index",
        chunks=chunks,
        indexed=chunks > 0,
    )


__all__ = ["app"]
