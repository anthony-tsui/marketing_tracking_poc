"""History-aware retrieval: rewrite, expand, search, rerank."""

from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Any, Sequence

from langchain_core.documents import Document

from martech_rag.config import Settings, get_settings
from martech_rag.ingest.indexer import similarity_search
from martech_rag.llm.openrouter import OpenRouterClient
from martech_rag.llm.schemas import (
    ChatMessage,
    ExpandedQueries,
    RerankResult,
    RetrievedChunk,
    RewrittenQuery,
)
from martech_rag.rag.prompts import (
    QUERY_EXPAND_PROMPT,
    QUERY_REWRITE_PROMPT,
    RERANK_PROMPT,
)

logger = logging.getLogger(__name__)

_STOPWORDS = {
    "the",
    "a",
    "an",
    "in",
    "to",
    "how",
    "do",
    "i",
    "and",
    "or",
    "of",
    "for",
    "using",
    "with",
    "on",
    "my",
    "is",
    "it",
    "from",
    "want",
    "need",
    "like",
    "please",
    "also",
    "track",
    "tracking",
    "use",
    "using",
    "get",
    "make",
    "can",
    "should",
}

# Page kinds — not topics. One set of rules for every question.
_KIND_MARKERS: dict[str, tuple[str, ...]] = {
    "api": (
        "protocol/ga4",
        "/rest/",
        "/config/admin",
        "reporting/data",
        "/api",
        "server-side/api",
        "tag-manager/api",
        "how-to-build-a-server-tag",
        "custom template permissions",
    ),
    "legacy": ("[ua]", "legacy", "analyticsjs"),
    "ads": (
        "floodlight",
        "campaign manager",
        "campaign-manager",
        "cm360",
        "dv360",
        "display and video",
        "fl-setup",
    ),
    "mobile": ("/android", "/ios/", "in-app purchase"),
    "warehouse": ("bigquery",),
    "plugin": (
        "webtoffee",
        "container size",
        "considerations before you install",
        "obtain user consent",
    ),
    "howto": (
        "set up",
        "set-up",
        "how to",
        "intro",
        "fundamentals",
        "measure ",
        "send data",
        "install",
        "configure",
        "/collection/ga4/",
        "sst-fundamentals",
        "recommended-events",
        "data layer",
        "datalayer",
        "enhanced measurement",
        "file download",
        "cloud-run",
        "custom-domain",
        "app-engine-setup",
        "tagging server",
    ),
}

_NOISE_KINDS = frozenset({"api", "legacy", "ads", "mobile", "warehouse", "plugin"})


def _query_terms(text: str) -> set[str]:
    return {
        t
        for t in re.findall(r"[a-z0-9]+", (text or "").lower())
        if len(t) > 1 and t not in _STOPWORDS
    }


def lexical_overlap(query: str, chunk_text: str) -> float:
    """Fraction of distinctive query terms that appear in the chunk."""
    terms = _query_terms(query)
    if not terms:
        return 0.0
    haystack = (chunk_text or "").lower()
    hits = sum(1 for t in terms if t in haystack)
    return hits / len(terms)


def page_kind(title: str, url: str, headers: str = "") -> str:
    blob = f"{title} {url} {headers}".lower()
    for kind, markers in _KIND_MARKERS.items():
        if any(marker in blob for marker in markers):
            return kind
    if "report" in (title or "").lower():
        return "report"
    return "other"


def query_intent(query: str) -> str:
    q = (query or "").lower()
    if any(token in q for token in ("bigquery", "data api", "admin api", "rest api")):
        return "warehouse" if "bigquery" in q else "api"
    if any(token in q for token in ("floodlight", "cm360", "dv360", "campaign manager")):
        return "ads"
    if any(token in q for token in ("android", "ios", "firebase")):
        return "mobile"
    if any(token in q for token in ("universal analytics", "[ua]", "analytics.js")):
        return "legacy"
    if any(token in q for token in ("report", "dashboard", "exploration", "explore")):
        return "report"
    return "howto"


def source_adjustment(query: str, title: str, url: str, headers: str = "") -> float:
    """Prefer how-to pages; downrank APIs, legacy, ads, and reports unless asked."""
    intent = query_intent(query)
    kind = page_kind(title, url, headers)
    if kind in _NOISE_KINDS and kind != intent:
        return -0.5
    if intent == "howto" and kind == "report":
        return -0.3
    if kind == intent or (intent == "howto" and kind == "howto"):
        return 0.4
    return 0.0


def search_variants(query: str) -> list[str]:
    """Search the question plus GA4/GTM phrasings built from its distinctive words."""
    q = (query or "").strip()
    if not q:
        return []
    ordered = [
        t
        for t in re.findall(r"[a-z0-9]+", q.lower())
        if t in _query_terms(q)
    ]
    core = " ".join(ordered) or q
    variants = [
        q,
        f"{core} Google Analytics 4",
        f"{core} Google Tag Manager",
    ]
    blob = f"{q} {core}".lower()
    if any(
        token in blob
        for token in ("server-side", "server side", "sgtm", "tagging server")
    ) or ("server" in ordered and "tagging" in ordered):
        variants.append(
            "Set up server-side tagging Cloud Run custom domain tagging server"
        )
    seen: list[str] = []
    for item in variants:
        if item and item not in seen:
            seen.append(item)
    return seen


def message_content(msg: ChatMessage | dict[str, Any] | Sequence[Any] | str | None) -> str:
    """Normalize Gradio / dict / ChatMessage content to plain text."""
    if msg is None:
        return ""
    if isinstance(msg, str):
        return msg.strip()
    if isinstance(msg, ChatMessage):
        return (msg.content or "").strip()
    if isinstance(msg, (list, tuple)):
        return " ".join(message_content(part) for part in msg).strip()
    if isinstance(msg, dict):
        content = msg.get("content", msg.get("text", ""))
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: list[str] = []
            for part in content:
                if isinstance(part, str):
                    parts.append(part)
                elif isinstance(part, dict):
                    parts.append(str(part.get("text") or part.get("content") or ""))
            return " ".join(parts).strip()
        return str(content or "").strip()
    return str(msg).strip()


def iter_turns(
    history: Sequence[ChatMessage] | Sequence[dict[str, Any]] | None,
) -> list[tuple[str, str]]:
    turns: list[tuple[str, str]] = []
    for msg in history or []:
        if isinstance(msg, (list, tuple)) and len(msg) >= 2:
            user, assistant = message_content(msg[0]), message_content(msg[1])
            if user:
                turns.append(("user", user))
            if assistant:
                turns.append(("assistant", assistant))
            continue
        if isinstance(msg, ChatMessage):
            turns.append((msg.role, message_content(msg)))
            continue
        if isinstance(msg, dict):
            role = str(msg.get("role") or "user")
            text = message_content(msg)
            if text:
                turns.append((role, text))
    return [(role, text) for role, text in turns if text]


def last_user_question(
    history: Sequence[ChatMessage] | Sequence[dict[str, Any]] | None,
) -> str:
    for role, text in reversed(iter_turns(history)):
        if role == "user":
            return text
    return ""


_FOLLOWUP_PREFIX = re.compile(
    r"^(what about|how about|and\b|also\b|via\b|i mean|but i mean|"
    r"instead|with\b|using\b|through\b|the same|ok but|yes but)",
    re.I,
)


def combine_followup(
    question: str,
    history: Sequence[ChatMessage] | Sequence[dict[str, Any]] | None,
) -> str:
    """Keep the prior event (e.g. purchase) when the user says 'what about SST'."""
    q = (question or "").strip()
    prev = last_user_question(history)
    if not prev:
        return q
    prev_terms = _query_terms(prev)
    q_terms = _query_terms(q)
    if _FOLLOWUP_PREFIX.search(q) or (prev_terms - q_terms and len(q_terms) <= 6):
        return f"{prev}. Follow-up: {q}"
    return q


def format_history(
    history: Sequence[ChatMessage] | Sequence[dict[str, Any]] | None,
) -> str:
    turns = iter_turns(history)
    if not turns:
        return "(none)"
    return "\n".join(f"{role}: {content}" for role, content in turns)


def docs_to_chunks(docs: list[Document]) -> list[RetrievedChunk]:
    chunks: list[RetrievedChunk] = []
    for doc in docs:
        meta = doc.metadata or {}
        chunks.append(
            RetrievedChunk(
                chunk_id=str(
                    meta.get("parent_id")
                    or meta.get("chunk_id")
                    or meta.get("id")
                    or hash(doc.page_content)
                ),
                content=str(meta.get("parent_text") or doc.page_content),
                score=float(meta.get("score") or 0.0),
                source_url=str(meta.get("source_url") or ""),
                title=str(meta.get("title") or ""),
                headers=str(meta.get("headers") or ""),
            )
        )
    return chunks


async def rewrite_query(
    question: str,
    history: Sequence[ChatMessage] | Sequence[dict[str, str]],
    client: OpenRouterClient,
    settings: Settings,
) -> str:
    try:
        result = await client.chat_structured(
            [
                {
                    "role": "user",
                    "content": QUERY_REWRITE_PROMPT.format(
                        history=format_history(history),
                        question=question,
                    ),
                }
            ],
            RewrittenQuery,
            model=settings.retrieval_model,
        )
        return result.standalone_query.strip() or question
    except Exception as exc:  # noqa: BLE001
        logger.warning("Query rewrite failed, using original question: %s", exc)
        return question


async def expand_query(
    query: str,
    client: OpenRouterClient,
    settings: Settings,
) -> list[str]:
    try:
        result = await client.chat_structured(
            [
                {
                    "role": "user",
                    "content": QUERY_EXPAND_PROMPT.format(
                        n=settings.expand_n,
                        query=query,
                    ),
                }
            ],
            ExpandedQueries,
            model=settings.retrieval_model,
        )
        queries = [q.strip() for q in result.queries if q.strip()]
        return queries[: settings.expand_n] or [query]
    except Exception as exc:  # noqa: BLE001
        logger.warning("Query expand failed, using original query: %s", exc)
        return [query]


async def rerank_chunks(
    query: str,
    history: Sequence[ChatMessage] | Sequence[dict[str, str]],
    chunks: list[RetrievedChunk],
    client: OpenRouterClient,
    settings: Settings,
) -> list[RetrievedChunk]:
    if not chunks:
        return []
    compact = []
    for c in chunks:
        snippet = c.content[:800]
        compact.append(f"- id={c.chunk_id}\n  title={c.title}\n  text={snippet}")
    try:
        result = await client.chat_structured(
            [
                {
                    "role": "user",
                    "content": RERANK_PROMPT.format(
                        history=format_history(history),
                        query=query,
                        chunks="\n".join(compact),
                    ),
                }
            ],
            RerankResult,
            model=settings.retrieval_model,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Rerank failed, using vector order: %s", exc)
        return chunks[: settings.rerank_top_n]
    by_id = {c.chunk_id: c for c in chunks}
    ranked: list[RetrievedChunk] = []
    for item in sorted(result.rankings, key=lambda x: x.relevance_score, reverse=True):
        chunk = by_id.get(item.chunk_id)
        if not chunk:
            continue
        ranked.append(
            RetrievedChunk(
                chunk_id=chunk.chunk_id,
                content=chunk.content,
                score=item.relevance_score,
                source_url=chunk.source_url,
                title=chunk.title,
                headers=chunk.headers,
            )
        )
    # Keep any missing chunks at the end with original scores
    seen = {c.chunk_id for c in ranked}
    for c in chunks:
        if c.chunk_id not in seen:
            ranked.append(c)
    return ranked[: settings.rerank_top_n]


async def retrieve(
    question: str,
    history: Sequence[ChatMessage] | Sequence[dict[str, str]] | None = None,
    *,
    settings: Settings | None = None,
    client: OpenRouterClient | None = None,
) -> tuple[str, list[RetrievedChunk]]:
    settings = settings or get_settings()
    history = history or []
    owns_client = client is None
    client = client or OpenRouterClient(settings)
    started = time.perf_counter()
    try:
        has_history = bool(iter_turns(history))
        rewritten = (
            combine_followup(question, history)
            if has_history
            else (question or "").strip()
        )
        extras = search_variants(rewritten)
        search_queries: list[str] = []
        for item in extras:
            if item and item not in search_queries:
                search_queries.append(item)

        merged: dict[str, RetrievedChunk] = {}
        results = await asyncio.gather(
            *[
                similarity_search(
                    q,
                    k=settings.retrieval_top_k,
                    settings=settings,
                    client=client,
                )
                for q in search_queries
            ]
        )
        for docs in results:
            for chunk in docs_to_chunks(docs):
                existing = merged.get(chunk.chunk_id)
                if existing is None or chunk.score > existing.score:
                    merged[chunk.chunk_id] = chunk

        candidates = sorted(merged.values(), key=lambda c: c.score, reverse=True)
        # Prefer chunks that mention the question terms (not just "GA4/GTM")
        for chunk in candidates:
            title_overlap = lexical_overlap(
                rewritten, f"{chunk.title} {chunk.source_url} {chunk.headers}"
            )
            body_overlap = lexical_overlap(rewritten, chunk.content)
            overlap = 0.65 * title_overlap + 0.35 * body_overlap
            boost = source_adjustment(
                rewritten, chunk.title, chunk.source_url, chunk.headers
            )
            # Don't boost a generic how-to page whose title never mentions the question
            if title_overlap < 0.4 and boost > 0:
                boost = 0.0
            chunk.score = 0.4 * chunk.score + 0.6 * overlap + boost
        ranked: list[RetrievedChunk] = []
        seen_urls: set[str] = set()
        for chunk in sorted(candidates, key=lambda c: c.score, reverse=True):
            url_key = chunk.source_url or chunk.chunk_id
            if url_key in seen_urls:
                continue
            seen_urls.add(url_key)
            ranked.append(chunk)
            if len(ranked) >= settings.rerank_top_n:
                break
        for chunk in ranked:
            chunk.score = max(0.0, min(1.0, chunk.score))
        logger.info(
            "Retrieve: rewritten=%r queries=%s kept=%s elapsed=%.1fs",
            rewritten,
            search_queries,
            len(ranked),
            time.perf_counter() - started,
        )
        return rewritten, ranked
    finally:
        if owns_client:
            await client.aclose()
