# Marketing Tracking RAG PoC

Local Python RAG over official **Google Analytics 4** and **Google Tag Manager** docs. Pages are crawled into Notion as **Web Clips**, indexed with LangChain markdown chunking + OpenRouter embeddings into **Chroma**, then answered in a marketer-facing **Gradio** chat with an eval dashboard.

Requires **Python 3.12**, `uv`, an OpenRouter key, and a Notion internal integration.

## Quick start

1. Clone the repo and install:

   ```bash
   uv sync
   ```

2. Copy env and fill secrets:

   ```bash
   cp .env.example .env
   ```

3. Put `OPENROUTER_API_KEY` and `NOTION_TOKEN` in `.env`. Share the Notes database, project, and Martech tag with the integration (see [Notion API setup](#notion-api-setup-required-once)).

4. Optional clean slate, then crawl → index → chat:

   ```bash
   uv run python scripts/reset_poc.py
   uv run python scripts/crawl_and_store.py
   uv run python scripts/index_from_notion.py
   uv run python scripts/run_app.py
   ```

Chat opens at [http://127.0.0.1:7860](http://127.0.0.1:7860) in Chrome when available.

Optional (only if you set `CRAWL_USE_BROWSER=true`):

```bash
uv run crawl4ai-setup
```

## Pipeline commands

| Command | Description |
|---------|-------------|
| `uv run python scripts/reset_poc.py` | Archive project Web Clips; clear crawl state, Chroma, eval artifacts |
| `uv run python scripts/reset_poc.py --keep-testset` | Same, but keep `data/eval/testset.json` |
| `uv run python scripts/crawl_and_store.py` | Recursive crawl + upsert Notion Web Clips (HTTP-only by default) |
| `uv run python scripts/crawl_and_store.py --max-pages 5 --reset-failed` | Smoke / retry failed or timed-out URLs |
| `uv run python scripts/repair_notion_labels.py` | Restore Project / Martech labels on clips that lost them |
| `uv run python scripts/index_from_notion.py` | Pull Notion notes → chunk → embed → Chroma (drops collection first) |
| `uv run python scripts/index_from_notion.py --no-reset` | Upsert into the existing collection |
| `uv run python scripts/generate_testset.py -n 10` | LLM-generate eval questions from indexed chunks |
| `uv run python scripts/run_eval.py --limit 5` | Judge answers (accuracy / completeness / relevance) |
| `uv run python scripts/run_app.py` | Gradio Chat + Eval UI |
| `uv run martech-rag-app` | Same Gradio app via the package entry point |

After crawl finishes, run `index_from_notion.py` before chat or eval.

## Features

- Recursive crawl of GA/GTM seed URLs on `developers.google.com` and `support.google.com`
  (only pages that mention Google Analytics or Google Tag Manager)
- Notion storage under project **Marketing tracking RAG PoC**, tag **Martech**, type **Web Clip**
- HTTP-only crawl by default (stable for Google docs); Playwright/Crawl4AI is optional
- Parent-child markdown chunking: embed short passages, return the parent section to the LLM
- Dedup of http/https twins and identical article bodies before indexing
- History-aware retrieval: follow-up rewrite, GA4/GTM search variants, vector search,
  lexical overlap, and page-kind ranking
- Async OpenRouter chat and embeddings; Pydantic structured outputs for eval / test-set generation
- Gradio **Chat** (retrieved chunks beside the answer) + **Eval** dashboard

## Models (OpenRouter)

| Role | Model | Used for |
|------|--------|----------|
| Chat | `deepseek/deepseek-v4-flash-0731` | Marketer-facing answers |
| Retrieval / eval | `z-ai/glm-5.2` | Test-set generation and LLM-as-judge (not live retrieval) |
| Embeddings | `nvidia/nemotron-3-embed-1b:free` | Index (`passage`) and search (`query`) |

Override with `CHAT_MODEL`, `RETRIEVAL_MODEL`, or `EMBED_MODEL` in `.env`.

## Architecture

```text
Seed URLs → HTTP crawl + GA/GTM filter → Notion Web Clips
                                              ↓
                         Notion API pull → dedupe → parent/child chunks
                                              ↓
                         OpenRouter embeddings → Chroma (martech_ga_gtm)
                                              ↓
                         Gradio Chat: follow-up + variants + vector + lexical rank
                         Gradio Eval: generate questions → answer → GLM judge
```

Key packages: `src/martech_rag/`

| Path | Role |
|------|------|
| `crawl/` | Recursive crawler and Notion Web Clip writer |
| `ingest/` | Notion reader, chunking, Chroma index |
| `rag/` | Retrieval, answer chain, prompts |
| `eval/` | Test set, judge, runner |
| `apps/gradio_app.py` | Chat + Eval UI |
| `config.py` | Settings from `.env` |

Live RAG techniques (parent-child chunks, hybrid ranking, what is unused): [docs/rag-techniques.md](docs/rag-techniques.md). Decisions: [tasks/plan.md](tasks/plan.md). Spec: [specs/rag-poc.md](specs/rag-poc.md).

## Notion API setup (required once)

1. Open [https://www.notion.so/my-integrations](https://www.notion.so/my-integrations)
2. Click **New integration** → copy the **Internal Integration Secret**
3. Put it in `.env` as `NOTION_TOKEN=...` (also set `OPENROUTER_API_KEY`)
4. Share with the integration (••• → Connect to):
   - Notes database: [https://app.notion.com/p/09e702c5d85382398eac018c27489af5](https://app.notion.com/p/09e702c5d85382398eac018c27489af5)
   - Project: Marketing tracking RAG PoC
   - Tag: Martech

IDs are already defaulted in `config.py` (`NOTION_NOTES_DATABASE_ID`, `NOTION_NOTES_DATA_SOURCE_ID`, `NOTION_PROJECT_ID`, `NOTION_MARTECH_TAG_ID`).

## Notes

- Crawl defaults to **HTTP-only**. Set `CRAWL_USE_BROWSER=true` to enable Playwright/Crawl4AI.
- Skips community/forum/profile/search URLs. Support pages are limited to Analytics / Tag Manager answer and topic paths.
- Extracts the main article body (strips nav/footer/chrome) before saving to Notion.
- Chat is written for **marketers briefing a developer**, not as a raw docs dump. Vague questions get clarifying questions first.
- Live retrieval does **not** call the unused LLM rewrite / expand / rerank helpers still in `retrieval.py`.
- Last local eval on 10 generated questions: accuracy **0.975**, completeness **0.985**, relevance **0.97**. Treat as a smoke signal, not a gold benchmark.
- Paths: crawl state `data/crawl_state.json`, Chroma `data/chroma/`, eval `data/eval/testset.json` and `data/eval/results.json`.
- `reset_poc.py` keeps non–Web Clip notes (for example Business requirements).
