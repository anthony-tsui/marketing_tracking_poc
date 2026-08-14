# Spec: Marketing Tracking RAG PoC

## Objective

Local Python RAG over official GA4/GTM docs stored in Notion, with Gradio chat + eval.

## Tech stack

Python 3.12, HTTP crawl (optional Crawl4AI/Playwright), Notion API, LangChain, Chroma, OpenRouter, Gradio, Pydantic.

## Commands

See [README.md](../README.md). Live RAG techniques: [docs/rag-techniques.md](../docs/rag-techniques.md).

## Success criteria

- Crawl stores GA/GTM pages as Notion Web Clips (Martech + project)
- Chroma index from Notion via parent-child markdown chunking + Nemotron embeddings
- Chat uses conversation follow-ups + retrieved context; UI shows chunks
- Answers are written for marketers briefing a developer
- Eval reports accuracy, completeness, relevance
- Async LLM/embed calls; structured outputs via Pydantic for eval and test-set generation

## Out of scope for this PoC

- Production hosting, auth, or multi-user access
- Using LLM rewrite / expand / rerank on the live retrieve path
- A human-authored gold eval set
