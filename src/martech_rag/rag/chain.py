"""RAG answer generation chain."""

from __future__ import annotations

import re
from typing import Sequence

from martech_rag.config import Settings, get_settings
from martech_rag.llm.openrouter import OpenRouterClient
from martech_rag.llm.schemas import ChatMessage, RagAnswer, RetrievedChunk
from martech_rag.rag.prompts import CHAT_USER_PROMPT, SYSTEM_PROMPT
from martech_rag.rag.retrieval import format_history, iter_turns, retrieve

_NAV_JUNK = re.compile(
    r"stay organized|save and categorize|skip to |was this helpful",
    re.I,
)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def quote_candidates(text: str, n: int = 2) -> list[str]:
    """Short lines the model can paste as blockquotes."""
    clean = " ".join((text or "").split())
    out: list[str] = []
    for part in _SENTENCE_SPLIT.split(clean):
        line = part.strip().strip('"“”')
        if len(line) < 40 or len(line) > 220:
            continue
        if _NAV_JUNK.search(line):
            continue
        out.append(line)
        if len(out) >= n:
            break
    return out


def format_context(chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return "(no retrieved context)"
    parts: list[str] = []
    for i, c in enumerate(chunks, 1):
        quotes = quote_candidates(c.content)
        quote_block = (
            "\n".join(f'> "{q}"\n> — [{c.title}]({c.source_url})' for q in quotes)
            if quotes
            else "(no short quote extracted — pick a sentence from the text)"
        )
        parts.append(
            f"[{i}] title={c.title}\n"
            f"url={c.source_url}\n"
            f"headers={c.headers}\n"
            f"score={c.score:.3f}\n"
            f"Ready-to-paste quotes:\n{quote_block}\n"
            f"Full text:\n{c.content}"
        )
    return "\n\n---\n\n".join(parts)


async def answer_question(
    question: str,
    history: Sequence[ChatMessage] | Sequence[dict[str, str]] | None = None,
    *,
    settings: Settings | None = None,
    client: OpenRouterClient | None = None,
) -> RagAnswer:
    settings = settings or get_settings()
    history = list(history or [])
    owns_client = client is None
    client = client or OpenRouterClient(settings)
    try:
        _rewritten, chunks = await retrieve(
            question,
            history,
            settings=settings,
            client=client,
        )
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for role, text in iter_turns(history)[-6:]:
            if role not in ("user", "assistant"):
                continue
            if role == "assistant" and len(text) > 1200:
                text = text[:1200] + "\n…"
            messages.append({"role": role, "content": text})
        messages.append(
            {
                "role": "user",
                "content": CHAT_USER_PROMPT.format(
                    history=format_history(history),
                    context=format_context(chunks),
                    question=question,
                ),
            }
        )
        answer = await client.chat(
            messages,
            model=settings.chat_model,
            temperature=0.3,
            max_tokens=8192,
        )
        return RagAnswer(answer=answer, chunks=chunks)
    finally:
        if owns_client:
            await client.aclose()
