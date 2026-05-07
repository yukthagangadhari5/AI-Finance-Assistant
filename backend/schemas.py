from __future__ import annotations

from typing import List

from pydantic import BaseModel, ConfigDict, Field


class QueryRequest(BaseModel):
    """Request payload for asking the assistant a question."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(
        ...,
        description="User's natural-language finance question to be answered by the assistant.",
        examples=[
            "Summarize my spending last month and identify the top 3 categories.",
            "How can I reduce my credit card interest based on my current balance?",
        ],
        min_length=1,
    )


class QueryResponse(BaseModel):
    """Response payload containing the assistant's answer and supporting metadata."""

    model_config = ConfigDict(extra="forbid")

    answer: str = Field(
        ...,
        description="Assistant's final answer in natural language.",
        examples=["Your largest spending category last month was Dining at $412.30..."],
        min_length=1,
    )
    sources: List[str] = Field(
        default_factory=list,
        description=(
            "List of source identifiers used to produce the answer (e.g., filenames, URLs, "
            "or internal document IDs)."
        ),
        examples=[["statements/jan_2026.pdf", "budget_rules.md"]],
    )
    confidence: float = Field(
        ...,
        description="Model confidence score for the answer, between 0 and 1.",
        ge=0.0,
        le=1.0,
        examples=[0.82],
    )
    tokens_used: int = Field(
        ...,
        description="Total tokens consumed to generate the response (prompt + completion).",
        ge=0,
        examples=[1534],
    )


class DocumentInfo(BaseModel):
    """Metadata about a document that may be indexed for retrieval."""

    model_config = ConfigDict(extra="forbid", from_attributes=True)

    filename: str = Field(
        ...,
        description="Document filename (or display name) as stored by the system.",
        examples=["statements/jan_2026.pdf"],
        min_length=1,
    )
    chunks: int = Field(
        ...,
        description="Number of text chunks created from this document for indexing.",
        ge=0,
        examples=[42],
    )
    indexed: bool = Field(
        ...,
        description="Whether this document is currently indexed and available for retrieval.",
        examples=[True],
    )

