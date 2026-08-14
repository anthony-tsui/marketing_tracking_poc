"""Gradio app: Chat + Eval tabs."""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import gradio as gr

from martech_rag.config import get_settings
from martech_rag.eval.runner import load_results, run_evaluation
from martech_rag.eval.testset import load_testset
from martech_rag.rag.chain import answer_question
from martech_rag.rag.retrieval import iter_turns

logger = logging.getLogger(__name__)


def _format_chunks(chunks: list[Any]) -> str:
    if not chunks:
        return "_No pages found for this question. The chat should say if it is not sure._"
    parts: list[str] = [
        "_Pages used for this answer. The chat states each fact in plain language, then quotes a short line from these pages._",
        "",
    ]
    for i, c in enumerate(chunks, 1):
        parts.append(
            f"### Chunk {i} (score={c.score:.3f})\n"
            f"**{c.title}**\n"
            f"URL: {c.source_url}\n"
            f"Headers: {c.headers or '-'}\n\n"
            f"{c.content[:1500]}"
        )
    return "\n\n---\n\n".join(parts)


def _history_dicts(history: list[Any] | None) -> list[dict[str, str]]:
    return [{"role": role, "content": text} for role, text in iter_turns(history)]


async def _chat_async(
    message: str,
    history: list[dict[str, str]],
) -> tuple[list[dict[str, str]], str]:
    history = _history_dicts(history)
    rag = await answer_question(message, history=history)
    new_history = [
        *history,
        {"role": "user", "content": message},
        {"role": "assistant", "content": rag.answer},
    ]
    return new_history, _format_chunks(rag.chunks)


def chat_fn(message: str, history: list[dict[str, str]]):
    if not message or not message.strip():
        return history or [], "_Ask a GA4/GTM question._"
    try:
        return asyncio.run(_chat_async(message.strip(), history or []))
    except Exception as exc:  # noqa: BLE001
        logger.exception("Chat failed")
        history = history or []
        err = (
            f"Sorry — chat failed: `{type(exc).__name__}: {exc}`. "
            "Check the terminal log. Retrieval will still try again on the next question."
        )
        return [
            *history,
            {"role": "user", "content": message.strip()},
            {"role": "assistant", "content": err},
        ], f"_Error: {exc}_"


def clear_chat():
    return [], "_Retrieved chunks will appear here._"


def run_eval_fn(limit: float):
    lim = int(limit) if limit and limit > 0 else None

    async def _run():
        return await run_evaluation(limit=lim)

    try:
        summary = asyncio.run(_run())
    except Exception as exc:  # noqa: BLE001
        logger.exception("Eval failed")
        return f"**Eval failed:** `{type(exc).__name__}: {exc}`", []
    rows = [
        [
            r.id,
            r.question,
            f"{r.accuracy:.2f}",
            f"{r.completeness:.2f}",
            f"{r.relevance:.2f}",
            r.rationale,
            r.chunk_count,
        ]
        for r in summary.rows
    ]
    metrics = (
        f"**Mean accuracy:** {summary.mean_accuracy:.3f}  \n"
        f"**Mean completeness:** {summary.mean_completeness:.3f}  \n"
        f"**Mean relevance:** {summary.mean_relevance:.3f}  \n"
        f"**Questions:** {len(summary.rows)}"
    )
    return metrics, rows


def load_saved_results():
    data = load_results()
    if not data:
        return "_No saved results. Run evaluation first._", []
    rows = [
        [
            r.get("id"),
            r.get("question"),
            f"{float(r.get('accuracy', 0)):.2f}",
            f"{float(r.get('completeness', 0)):.2f}",
            f"{float(r.get('relevance', 0)):.2f}",
            r.get("rationale", ""),
            r.get("chunk_count", 0),
        ]
        for r in data.get("rows", [])
    ]
    metrics = (
        f"**Mean accuracy:** {data.get('mean_accuracy', 0):.3f}  \n"
        f"**Mean completeness:** {data.get('mean_completeness', 0):.3f}  \n"
        f"**Mean relevance:** {data.get('mean_relevance', 0):.3f}  \n"
        f"**Questions:** {len(rows)}"
    )
    return metrics, rows


def testset_info() -> str:
    settings = get_settings()
    path = settings.eval_testset_path
    if not path.exists():
        return f"No test set at `{path}`. Run `uv run python scripts/generate_testset.py`."
    ts = load_testset(settings=settings)
    return f"Loaded **{len(ts.items)}** questions from `{path}`."


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Marketing Tracking RAG PoC") as demo:
        gr.Markdown(
            "# Marketing Tracking RAG PoC\n"
            "Ask GA4 / GTM questions in plain language. "
            "Answers are written for **marketers** who need to brief a developer."
        )
        with gr.Tab("Chat"):
            with gr.Row():
                with gr.Column(scale=2):
                    chatbot = gr.Chatbot(height=480, label="GA4 / GTM guide")
                    msg = gr.Textbox(
                        label="Question",
                        placeholder="e.g. Count newsletter signups after the form really submits",
                    )
                    with gr.Row():
                        send = gr.Button("Send", variant="primary")
                        clear = gr.Button("Clear")
                with gr.Column(scale=1):
                    chunks_md = gr.Markdown(
                        "_Retrieved chunks will appear here._",
                        label="Retrieved chunks",
                        height=480,
                    )

            send.click(chat_fn, inputs=[msg, chatbot], outputs=[chatbot, chunks_md]).then(
                lambda: "", outputs=msg
            )
            msg.submit(chat_fn, inputs=[msg, chatbot], outputs=[chatbot, chunks_md]).then(
                lambda: "", outputs=msg
            )
            clear.click(clear_chat, outputs=[chatbot, chunks_md])

        with gr.Tab("Eval"):
            gr.Markdown("Evaluate RAG **accuracy**, **completeness**, and **relevance**.")
            info = gr.Markdown(testset_info())
            limit = gr.Number(value=5, label="Limit (0 = all)", precision=0)
            with gr.Row():
                run_btn = gr.Button("Run evaluation", variant="primary")
                load_btn = gr.Button("Load saved results")
            metrics = gr.Markdown("_Metrics will appear here._")
            table = gr.Dataframe(
                headers=[
                    "id",
                    "question",
                    "accuracy",
                    "completeness",
                    "relevance",
                    "rationale",
                    "chunks",
                ],
                datatype=["str", "str", "str", "str", "str", "str", "number"],
                interactive=False,
                wrap=True,
            )
            run_btn.click(run_eval_fn, inputs=[limit], outputs=[metrics, table])
            load_btn.click(load_saved_results, outputs=[metrics, table])
            demo.load(testset_info, outputs=info)

    return demo


def _open_in_chrome(url: str) -> None:
    """Open the Gradio URL in Google Chrome (not Cursor's Simple Browser)."""
    candidates = [
        Path(os.environ.get("PROGRAMFILES", r"C:\Program Files"))
        / "Google/Chrome/Application/chrome.exe",
        Path(os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)"))
        / "Google/Chrome/Application/chrome.exe",
        Path(os.environ.get("LOCALAPPDATA", ""))
        / "Google/Chrome/Application/chrome.exe",
    ]
    for chrome in candidates:
        if chrome.is_file():
            subprocess.Popen([str(chrome), url], close_fds=True)
            return
    logger.warning("Chrome not found. Open this URL manually: %s", url)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    # Skip Gradio/HF telemetry pings that log a harmless WinError 10054 on Windows
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
    app = build_app()
    threading.Thread(
        target=lambda: (time.sleep(1.2), _open_in_chrome("http://127.0.0.1:7860")),
        daemon=True,
    ).start()
    app.launch(
        inbrowser=False,
        server_name="127.0.0.1",
        server_port=7860,
    )


if __name__ == "__main__":
    main()
