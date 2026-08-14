"""Crawl official GA/GTM docs and upsert into Notion as Web Clips."""

from __future__ import annotations

import argparse
import asyncio
import logging

from martech_rag.config import get_settings
from martech_rag.crawl.crawler import crawl_docs
from martech_rag.crawl.notion_writer import NotionWriter


async def main_async(max_pages: int | None, reset_failed: bool) -> None:
    settings = get_settings()
    writer = NotionWriter(settings)
    stored = 0

    async def on_page(page):
        nonlocal stored
        await writer.upsert_page(page)
        stored += 1
        logging.info("Stored %s (%s total this run)", page.url, stored)

    try:
        pages = await crawl_docs(
            settings=settings,
            on_page=on_page,
            max_pages=max_pages,
            reset_failed=reset_failed,
        )
        logging.info("Crawl finished. Kept pages this run: %s", len(pages))
    finally:
        await writer.aclose()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description="Crawl GA/GTM docs into Notion")
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Optional cap on pages stored this run (for smoke tests)",
    )
    parser.add_argument(
        "--reset-failed",
        action="store_true",
        help="Retry URLs previously marked failed (e.g. anti-bot blocks)",
    )
    args = parser.parse_args()
    asyncio.run(main_async(args.max_pages, args.reset_failed))


if __name__ == "__main__":
    main()
