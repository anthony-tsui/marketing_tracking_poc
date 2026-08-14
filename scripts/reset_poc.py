"""Reset PoC to a clean re-runnable state.

- Archives all project Web Clips in Notion
- Clears crawl state, Chroma index, and eval artifacts
Does NOT delete non-Web-Clip notes (e.g. Business requirements).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import shutil
from pathlib import Path

from martech_rag.config import get_settings
from martech_rag.notion_api import make_async_client, notion_call

logger = logging.getLogger(__name__)


async def archive_web_clips() -> int:
    settings = get_settings()
    client = make_async_client(settings)
    archived = 0
    try:
        cursor = None
        while True:
            kwargs = {
                "data_source_id": settings.notion_notes_data_source_id,
                "filter": {
                    "and": [
                        {
                            "property": "Project",
                            "relation": {"contains": settings.notion_project_id},
                        },
                        {"property": "Type", "select": {"equals": "Web Clip"}},
                        {"property": "Archived", "checkbox": {"equals": False}},
                    ]
                },
                "page_size": 100,
            }
            if cursor:
                kwargs["start_cursor"] = cursor
            resp = await notion_call(
                lambda: client.data_sources.query(**kwargs),
                what="list web clips to archive",
            )
            results = resp.get("results") or []
            for page in results:
                await notion_call(
                    lambda pid=page["id"]: client.pages.update(
                        page_id=pid, archived=True
                    ),
                    what=f"archive {page['id']}",
                )
                # Also flip Ultimate Brain Archived checkbox when present
                try:
                    await notion_call(
                        lambda pid=page["id"]: client.pages.update(
                            page_id=pid,
                            properties={"Archived": {"checkbox": True}},
                        ),
                        what=f"archive checkbox {page['id']}",
                    )
                except Exception:  # noqa: BLE001
                    pass
                archived += 1
            if not resp.get("has_more"):
                break
            cursor = resp.get("next_cursor")
    finally:
        await client.aclose()
    return archived


def clear_local_state(*, clear_testset: bool) -> None:
    settings = get_settings()
    state = Path(settings.crawl_state_path)
    if state.exists():
        state.unlink()
        logger.info("Removed %s", state)

    chroma = Path(settings.chroma_dir)
    if chroma.exists():
        shutil.rmtree(chroma, ignore_errors=True)
        chroma.mkdir(parents=True, exist_ok=True)
        logger.info("Cleared Chroma at %s", chroma)

    eval_dir = Path(settings.eval_testset_path).parent
    results = eval_dir / "results.json"
    if results.exists():
        results.unlink()
        logger.info("Removed %s", results)
    if clear_testset and settings.eval_testset_path.exists():
        settings.eval_testset_path.unlink()
        logger.info("Removed %s", settings.eval_testset_path)


async def main_async(clear_testset: bool) -> None:
    n = await archive_web_clips()
    logger.info("Archived %s active Web Clips", n)
    clear_local_state(clear_testset=clear_testset)
    logger.info("PoC is clean. Re-run with scripts/crawl_and_store.py")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Reset Marketing Tracking RAG PoC")
    parser.add_argument(
        "--keep-testset",
        action="store_true",
        help="Keep data/eval/testset.json",
    )
    args = parser.parse_args()
    asyncio.run(main_async(clear_testset=not args.keep_testset))


if __name__ == "__main__":
    main()
