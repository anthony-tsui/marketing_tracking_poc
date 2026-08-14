"""Generate and load evaluation test sets."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from martech_rag.config import Settings, get_settings
from martech_rag.ingest.indexer import get_chroma_collection
from martech_rag.llm.openrouter import OpenRouterClient
from martech_rag.llm.schemas import GeneratedTestItems, TestSet
from martech_rag.rag.prompts import TESTSET_GEN_PROMPT

logger = logging.getLogger(__name__)


def load_testset(path: Path | None = None, settings: Settings | None = None) -> TestSet:
    settings = settings or get_settings()
    path = path or settings.eval_testset_path
    data = json.loads(path.read_text(encoding="utf-8"))
    return TestSet.model_validate(data)


def save_testset(testset: TestSet, path: Path | None = None, settings: Settings | None = None) -> Path:
    settings = settings or get_settings()
    path = path or settings.eval_testset_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(testset.model_dump(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


async def generate_testset(
    *,
    n: int = 10,
    settings: Settings | None = None,
) -> TestSet:
    settings = settings or get_settings()
    collection = get_chroma_collection(settings)
    total = collection.count()
    if total == 0:
        raise RuntimeError("Chroma is empty. Run index_from_notion first.")

    # Sample a spread of documents for question generation
    sample_size = min(20, total)
    raw = collection.get(limit=sample_size, include=["documents", "metadatas"])
    docs = raw.get("documents") or []
    metas = raw.get("metadatas") or []
    excerpts: list[str] = []
    for i, (doc, meta) in enumerate(zip(docs, metas)):
        meta = meta or {}
        excerpts.append(
            f"Excerpt {i+1}\n"
            f"title={meta.get('title')}\n"
            f"url={meta.get('source_url')}\n"
            f"{(doc or '')[:1200]}"
        )

    async with OpenRouterClient(settings) as client:
        generated = await client.chat_structured(
            [
                {
                    "role": "user",
                    "content": TESTSET_GEN_PROMPT.format(
                        n=n,
                        excerpts="\n\n".join(excerpts),
                    ),
                }
            ],
            GeneratedTestItems,
            model=settings.retrieval_model,
        )

    # Normalize ids
    items = []
    for i, item in enumerate(generated.items[:n], 1):
        item.id = item.id or f"q{i:02d}"
        items.append(item)
    testset = TestSet(items=items)
    save_testset(testset, settings=settings)
    logger.info("Wrote test set with %s items to %s", len(items), settings.eval_testset_path)
    return testset
