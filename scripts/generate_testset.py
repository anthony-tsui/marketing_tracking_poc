"""Generate eval test set from indexed chunks."""

from __future__ import annotations

import argparse
import asyncio
import logging

from martech_rag.eval.testset import generate_testset


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=10, help="Number of questions")
    args = parser.parse_args()
    testset = asyncio.run(generate_testset(n=args.n))
    print(f"Generated {len(testset.items)} questions")


if __name__ == "__main__":
    main()
