"""Markdown-aware document chunking."""

from __future__ import annotations

import hashlib
import logging
import re
from urllib.parse import urlparse, urlunparse

from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from martech_rag.config import Settings, get_settings
from martech_rag.ingest.notion_reader import NotionNote

logger = logging.getLogger(__name__)

_SOURCE_LINE = re.compile(r"^source:\s*\S+", re.IGNORECASE)
_WHITESPACE = re.compile(r"\s+")


def _canonical_url_key(url: str) -> str:
    """http/https and trailing slash should not create extra notes."""
    parsed = urlparse((url or "").strip())
    host = parsed.netloc.lower()
    path = (parsed.path or "/").rstrip("/") or "/"
    query = parsed.query
    if "support.google.com" in host:
        parts = [p for p in query.split("&") if p.startswith("hl=")]
        query = "&".join(parts) if parts else "hl=en"
    else:
        query = ""
    return urlunparse(("https", host, path, "", query, ""))


def _body_fingerprint(markdown: str, title: str = "") -> str:
    """Hash article text only — ignore injected title/Source URL lines."""
    parts: list[str] = []
    title_norm = (title or "").strip().lower()
    for raw in markdown.splitlines():
        line = raw.strip()
        if not line or _SOURCE_LINE.match(line):
            continue
        if line.startswith("#"):
            heading = line.lstrip("#").strip().lower()
            if title_norm and heading == title_norm:
                continue
            line = heading
        parts.append(line.lower())
    body = _WHITESPACE.sub(" ", " ".join(parts)).strip()
    if not body:
        return ""
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _note_rank(note: NotionNote) -> tuple[int, int, int]:
    url = note.url or ""
    path = urlparse(url).path.lower()
    return (
        1 if url.startswith("https://") else 0,
        1 if "/answer/" in path else 0,
        len(note.markdown or ""),
    )


def _prefer(current: NotionNote, candidate: NotionNote) -> NotionNote:
    return candidate if _note_rank(candidate) > _note_rank(current) else current


def dedupe_notes(notes: list[NotionNote]) -> list[NotionNote]:
    """Collapse http/https twins and identical article bodies before chunking."""
    by_url: dict[str, NotionNote] = {}
    for note in notes:
        key = _canonical_url_key(note.url) if note.url else note.page_id
        existing = by_url.get(key)
        by_url[key] = _prefer(existing, note) if existing else note

    by_body: dict[str, NotionNote] = {}
    skipped_empty = 0
    for note in by_url.values():
        fingerprint = _body_fingerprint(note.markdown, note.title)
        if not fingerprint:
            skipped_empty += 1
            continue
        existing = by_body.get(fingerprint)
        by_body[fingerprint] = _prefer(existing, note) if existing else note

    kept = list(by_body.values())
    dropped = len(notes) - len(kept)
    logger.info(
        "Deduped notes before chunking: %s -> %s (dropped %s, empty %s)",
        len(notes),
        len(kept),
        dropped,
        skipped_empty,
    )
    return kept


_NUMBERED_STEP = re.compile(r"(?m)(?=^\d+\.\s)")
_BOILERPLATE_PHRASES = (
    "configure the google tag",
    "enables data to flow from your website",
    "unique google tag for each website",
    "set up google analytics in tag manager",
    "google tag enables data to flow",
)
_PROCEDURE_HINTS = (
    "click",
    "select",
    "trigger",
    "event",
    "variable",
    "preview",
    "publish",
    "button",
    "tag type",
    "measurement id",
    "gtm-",
    "g-",
)


def _pack_parts(parts: list[str], max_chars: int) -> list[str]:
    buckets: list[str] = []
    buf = ""
    for part in parts:
        piece = part.strip()
        if not piece:
            continue
        if buf and len(buf) + 2 + len(piece) > max_chars:
            buckets.append(buf)
            buf = piece
        else:
            buf = f"{buf}\n\n{piece}" if buf else piece
    if buf:
        buckets.append(buf)
    return buckets


def _split_section_into_parents(text: str, max_chars: int) -> list[str]:
    """Keep how-to steps together; split long articles on numbered lists."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    steps = [p.strip() for p in _NUMBERED_STEP.split(text) if p.strip()]
    if len(steps) > 1:
        return _pack_parts(steps, max_chars)
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(paras) > 1:
        return _pack_parts(paras, max_chars)
    splitter = RecursiveCharacterTextSplitter(chunk_size=max_chars, chunk_overlap=120)
    return [d.page_content.strip() for d in splitter.create_documents([text]) if d.page_content.strip()]


def _is_boilerplate(text: str) -> bool:
    """Drop generic 'set up the Google tag' lead-ins that drown real how-to chunks."""
    compact = _WHITESPACE.sub(" ", (text or "").lower())
    if len(compact) < 60:
        return True
    boiler_hits = sum(1 for phrase in _BOILERPLATE_PHRASES if phrase in compact)
    has_procedure = any(hint in compact for hint in _PROCEDURE_HINTS)
    if boiler_hits >= 2 and not has_procedure:
        return True
    if boiler_hits >= 2 and len(compact) < 700:
        return True
    return False


def _heading_line(meta: dict, title: str) -> str:
    for key in ("h3", "h2", "h1"):
        value = (meta.get(key) or "").strip()
        if value:
            return value
    return (title or "").strip()


def chunk_notion_notes(
    notes: list[NotionNote],
    settings: Settings | None = None,
) -> list[Document]:
    """Parent-child chunks: embed small passages, keep the full section for the LLM.

    Google help pages often start with the same intro paragraph. Indexing that
    1200-char lead made every query retrieve 'set up the Google tag' instead of
    the later click / trigger / event steps.
    """
    settings = settings or get_settings()
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "h1"),
            ("##", "h2"),
            ("###", "h3"),
        ],
        strip_headers=False,
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    docs: list[Document] = []
    dropped_boilerplate = 0
    for note in notes:
        text = note.markdown.strip()
        if not text:
            continue
        if not text.lstrip().startswith("#"):
            text = f"# {note.title}\n\n{text}"

        sections = header_splitter.split_text(text)
        if not sections:
            sections = [Document(page_content=text, metadata={})]

        section_idx = 0
        for section in sections:
            headers = " > ".join(
                filter(
                    None,
                    [
                        section.metadata.get("h1"),
                        section.metadata.get("h2"),
                        section.metadata.get("h3"),
                    ],
                )
            )
            heading = _heading_line(section.metadata, note.title)
            parents = _split_section_into_parents(
                section.page_content,
                settings.parent_chunk_size,
            )
            for parent_text in parents:
                if _is_boilerplate(parent_text):
                    dropped_boilerplate += 1
                    continue
                parent_id = f"{note.page_id}:p{section_idx}"
                section_idx += 1
                stored_parent = parent_text[: settings.parent_chunk_size]
                children = child_splitter.split_text(parent_text)
                if not children:
                    children = [parent_text]
                child_idx = 0
                for child in children:
                    child = child.strip()
                    if len(child) < 80 or _is_boilerplate(child):
                        dropped_boilerplate += 1
                        continue
                    # Embed the specific heading + passage (not only the generic page title)
                    embed_text = f"{heading}\n\n{child}" if heading else child
                    chunk_id = (
                        f"{parent_id}:{child_idx}:"
                        f"{hash(embed_text) & 0xFFFFFFFF:x}"
                    )
                    child_idx += 1
                    docs.append(
                        Document(
                            page_content=embed_text,
                            metadata={
                                "chunk_id": chunk_id,
                                "parent_id": parent_id,
                                "parent_text": stored_parent,
                                "notion_page_id": note.page_id,
                                "source_url": note.url,
                                "title": note.title,
                                "headers": headers or heading,
                            },
                        )
                    )
    logger.info(
        "Chunked %s notes into %s search passages (dropped %s boilerplate intros)",
        len(notes),
        len(docs),
        dropped_boilerplate,
    )
    return docs
