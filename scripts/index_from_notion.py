"""Index Notion Web Clips into Chroma."""

from __future__ import annotations

import argparse
import asyncio
import logging

from martech_rag.ingest.indexer import index_from_notion


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-reset", action="store_true", help="Do not drop existing collection")
    args = parser.parse_args()
    count = asyncio.run(index_from_notion(reset=not args.no_reset))
    print(f"Indexed {count} chunks")


if __name__ == "__main__":
    main()
