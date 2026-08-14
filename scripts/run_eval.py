"""CLI runner for RAG evaluation."""

from __future__ import annotations

import argparse
import asyncio
import logging

from martech_rag.eval.runner import run_evaluation


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    summary = asyncio.run(run_evaluation(limit=args.limit))
    print(
        f"accuracy={summary.mean_accuracy:.3f} "
        f"completeness={summary.mean_completeness:.3f} "
        f"relevance={summary.mean_relevance:.3f} "
        f"n={len(summary.rows)}"
    )


if __name__ == "__main__":
    main()
