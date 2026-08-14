"""Repair crawled Notion Web Clips missing Project and/or Martech tag."""

from __future__ import annotations

import asyncio
import logging

from martech_rag.crawl.notion_writer import NotionWriter


async def main_async() -> None:
    writer = NotionWriter()
    try:
        stats = await writer.repair_missing_labels()
        print(f"checked={stats['checked']} fixed={stats['fixed']}")
    finally:
        await writer.aclose()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
