from __future__ import annotations

import os
from typing import Any, Dict, List, Literal, TypedDict

import google.generativeai as genai
from langgraph.graph import END, START, StateGraph
from dotenv import load_dotenv

from backend.retriever import search as retrieve_search

QueryType = Literal["factual", "analytical", "comparison"]


class AgentState(TypedDict, total=False):
    query: str
    query_type: QueryType
    chunks: List[Dict[str, Any]]
    answer: str
    sources: List[str]
    confidence: float
    tokens_used: int


def router_agent(state: AgentState) -> AgentState:
    """
    Classify query as: factual, analytical, or comparison using keyword matching.
    """
    query = (state.get("query") or "").strip()
    q = query.lower()

    comparison_keywords = [
        "compare",
        "comparison",
        "vs",
        "versus",
        "difference",
        "differences",
        "better",
        "which is better",
        "pros and cons",
        "pros",
        "cons",
    ]
    analytical_keywords = [
        "analyze",
        "analysis",
        "trend",
        "breakdown",
        "summary",
        "summarize",
        "insights",
        "why",
        "how",
        "calculate",
        "forecast",
        "projection",
        "correlation",
        "optimize",
        "reduce",
        "increase",
        "budget",
        "spending",
        "category",
        "categories",
        "top",
        "average",
        "median",
        "variance",
    ]

    if any(kw in q for kw in comparison_keywords):
        query_type: QueryType = "comparison"
    elif any(kw in q for kw in analytical_keywords):
        query_type = "analytical"
    else:
        query_type = "factual"

    return {**state, "query": query, "query_type": query_type}


def _scores_to_confidence(scores: List[float]) -> float:
    """
    Convert FAISS distance-like scores (lower is better) to a 0..1 confidence.

    Heuristic: similarity_i = 1 / (1 + score_i), confidence = mean(similarity_i).
    """
    if not scores:
        return 0.0
    sims: List[float] = []
    for s in scores:
        try:
            s_float = float(s)
        except Exception:
            continue
        sims.append(1.0 / (1.0 + max(0.0, s_float)))
    if not sims:
        return 0.0
    conf = sum(sims) / len(sims)
    if conf < 0.0:
        return 0.0
    if conf > 1.0:
        return 1.0
    return conf


def research_agent(state: AgentState) -> AgentState:
    """
    Retrieve relevant context and answer using Google Gemini (gemini-1.5-flash).

    Returns: answer, sources, confidence, tokens_used (and also includes chunks).
    """
    query = (state.get("query") or "").strip()
    query_type: QueryType = state.get("query_type") or "factual"

    chunks = retrieve_search(query, k=3)
    sources = sorted({str(c.get("source", "unknown")) for c in chunks if c})
    scores = [float(c.get("score", 0.0)) for c in chunks if c and "score" in c]
    confidence = _scores_to_confidence(scores)

    context_lines: List[str] = []
    for i, c in enumerate(chunks, start=1):
        context_lines.append(
            f"[{i}] Source: {c.get('source','unknown')}\n"
            f"Score: {c.get('score')}\n"
            f"Chunk:\n{c.get('chunk','')}\n"
        )
    context = "\n---\n".join(context_lines) if context_lines else "(no retrieved context)"

    system_prompt = (
        "You are a helpful finance assistant. Answer using only the provided context. "
        "If the context is insufficient, say what is missing and make a best-effort "
        "answer without inventing facts."
    )
    user_prompt = (
        f"Query type: {query_type}\n\n"
        f"User question:\n{query}\n\n"
        f"Retrieved context:\n{context}\n\n"
        "Write a concise, well-structured answer. Cite sources by referring to [1], [2], [3] "
        "when you use information from a chunk."
    )

    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY is not set. Add it to your .env file or environment variables."
        )

    genai.configure(api_key=api_key)
    # `google-generativeai` expects full resource names from `list_models()`.
    model = genai.GenerativeModel(
        model_name="models/gemini-flash-latest",
        system_instruction=system_prompt,
    )
    response = model.generate_content(
        user_prompt,
        generation_config=genai.GenerationConfig(temperature=0.2),
    )

    answer = (getattr(response, "text", None) or "").strip()
    usage = getattr(response, "usage_metadata", None)
    tokens_used = int(getattr(usage, "total_token_count", 0) or 0)

    return {
        **state,
        "chunks": chunks,
        "answer": answer,
        "sources": sources,
        "confidence": float(confidence),
        "tokens_used": int(tokens_used),
    }


def _build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("router_agent", router_agent)
    graph.add_node("research_agent", research_agent)

    graph.add_edge(START, "router_agent")
    graph.add_edge("router_agent", "research_agent")
    graph.add_edge("research_agent", END)
    return graph.compile()


_GRAPH = _build_graph()


def run_query(question: str) -> AgentState:
    """
    Run the full LangGraph pipeline and return final state.
    """
    initial: AgentState = {"query": question}
    return _GRAPH.invoke(initial)


__all__ = ["run_query", "router_agent", "research_agent", "AgentState"]
