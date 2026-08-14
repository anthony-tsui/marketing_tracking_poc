"""LLM-as-judge for RAG answers."""

from __future__ import annotations

from martech_rag.config import Settings, get_settings
from martech_rag.llm.openrouter import OpenRouterClient
from martech_rag.llm.schemas import EvalScores, RetrievedChunk, TestQuestion
from martech_rag.rag.chain import format_context
from martech_rag.rag.prompts import EVAL_JUDGE_PROMPT


async def judge_answer(
    question: TestQuestion,
    answer: str,
    chunks: list[RetrievedChunk],
    *,
    settings: Settings | None = None,
    client: OpenRouterClient | None = None,
) -> EvalScores:
    settings = settings or get_settings()
    owns_client = client is None
    client = client or OpenRouterClient(settings)
    try:
        return await client.chat_structured(
            [
                {
                    "role": "user",
                    "content": EVAL_JUDGE_PROMPT.format(
                        question=question.question,
                        expected=question.expected_answer,
                        key_points="\n".join(f"- {p}" for p in question.key_points),
                        answer=answer,
                        context=format_context(chunks)[:6000],
                    ),
                }
            ],
            EvalScores,
            model=settings.retrieval_model,
        )
    finally:
        if owns_client:
            await client.aclose()
