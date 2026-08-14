"""Run evaluation over the test set."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

from martech_rag.config import Settings, get_settings
from martech_rag.eval.judge import judge_answer
from martech_rag.eval.testset import load_testset
from martech_rag.llm.openrouter import OpenRouterClient
from martech_rag.rag.chain import answer_question

logger = logging.getLogger(__name__)


@dataclass
class EvalRow:
    id: str
    question: str
    answer: str
    accuracy: float
    completeness: float
    relevance: float
    rationale: str
    chunk_count: int
    source_urls: list[str]


@dataclass
class EvalSummary:
    rows: list[EvalRow]
    mean_accuracy: float
    mean_completeness: float
    mean_relevance: float


async def run_evaluation(
    *,
    settings: Settings | None = None,
    limit: int | None = None,
) -> EvalSummary:
    settings = settings or get_settings()
    testset = load_testset(settings=settings)
    items = testset.items[:limit] if limit else testset.items
    rows: list[EvalRow] = []

    async with OpenRouterClient(settings) as client:
        for item in items:
            logger.info("Evaluating %s", item.id)
            rag = await answer_question(item.question, history=[], settings=settings, client=client)
            scores = await judge_answer(
                item,
                rag.answer,
                rag.chunks,
                settings=settings,
                client=client,
            )
            rows.append(
                EvalRow(
                    id=item.id,
                    question=item.question,
                    answer=rag.answer,
                    accuracy=scores.accuracy,
                    completeness=scores.completeness,
                    relevance=scores.relevance,
                    rationale=scores.rationale,
                    chunk_count=len(rag.chunks),
                    source_urls=[c.source_url for c in rag.chunks if c.source_url],
                )
            )

    def mean(vals: list[float]) -> float:
        return sum(vals) / len(vals) if vals else 0.0

    summary = EvalSummary(
        rows=rows,
        mean_accuracy=mean([r.accuracy for r in rows]),
        mean_completeness=mean([r.completeness for r in rows]),
        mean_relevance=mean([r.relevance for r in rows]),
    )
    out = settings.eval_testset_path.parent / "results.json"
    out.write_text(
        json.dumps(
            {
                "mean_accuracy": summary.mean_accuracy,
                "mean_completeness": summary.mean_completeness,
                "mean_relevance": summary.mean_relevance,
                "rows": [asdict(r) for r in summary.rows],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    logger.info(
        "Eval means accuracy=%.3f completeness=%.3f relevance=%.3f -> %s",
        summary.mean_accuracy,
        summary.mean_completeness,
        summary.mean_relevance,
        out,
    )
    return summary


def load_results(path: Path | None = None, settings: Settings | None = None) -> dict:
    settings = settings or get_settings()
    path = path or (settings.eval_testset_path.parent / "results.json")
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
