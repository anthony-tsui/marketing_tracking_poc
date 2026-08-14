"""Pydantic schemas for structured LLM outputs."""

from __future__ import annotations

from pydantic import BaseModel, Field


class RewrittenQuery(BaseModel):
    standalone_query: str = Field(
        ...,
        description="Standalone search query rewritten with conversation history.",
    )


class ExpandedQueries(BaseModel):
    queries: list[str] = Field(
        ...,
        min_length=1,
        description="Alternative search queries that cover related phrasings.",
    )


class RerankItem(BaseModel):
    chunk_id: str
    relevance_score: float = Field(..., ge=0.0, le=1.0)
    reason: str = ""


class RerankResult(BaseModel):
    rankings: list[RerankItem]


class EvalScores(BaseModel):
    accuracy: float = Field(..., ge=0.0, le=1.0)
    completeness: float = Field(..., ge=0.0, le=1.0)
    relevance: float = Field(..., ge=0.0, le=1.0)
    rationale: str = ""


class TestQuestion(BaseModel):
    id: str
    question: str
    expected_answer: str
    key_points: list[str] = Field(default_factory=list)
    gold_source_urls: list[str] = Field(default_factory=list)


class TestSet(BaseModel):
    items: list[TestQuestion]


class GeneratedTestItems(BaseModel):
    items: list[TestQuestion]


class ChatMessage(BaseModel):
    role: str
    content: str


class RetrievedChunk(BaseModel):
    chunk_id: str
    content: str
    score: float = 0.0
    source_url: str = ""
    title: str = ""
    headers: str = ""


class RagAnswer(BaseModel):
    answer: str
    chunks: list[RetrievedChunk] = Field(default_factory=list)
