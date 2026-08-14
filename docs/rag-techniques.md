# RAG techniques used in this PoC

This is the **live** pipeline as of August 2026. Chat retrieval does **not** call LLM query rewrite, query expansion, or LLM rerank. Those helpers still exist in `src/martech_rag/rag/retrieval.py` but `retrieve()` never uses them.

Code map: indexing in `ingest/chunking.py` and `ingest/indexer.py`; retrieval in `rag/retrieval.py`; answers in `rag/chain.py` and `rag/prompts.py`.

## Everyday picture

Think of Google’s help site as a huge filing cabinet.

1. **Index** — cut each article into small labeled slips (children) and keep the full section on the back of the slip (parent).
2. **Retrieve** — when a marketer asks a question, search several phrasings of that question, then prefer slips whose titles actually mention the thing they asked about (purchase, file download, tagging server), not generic “set up the Google tag” pages.
3. **Answer** — give the model those sections plus recent chat turns, and ask it to brief a marketer who will copy instructions to a developer.

Example: “I want to track purchases.” A naive vector search often returns Floodlight / Campaign Manager pages because they also say “purchase”. Lexical overlap + page-kind ranking push those down and keep GA4 ecommerce / data layer how-tos.

## Indexing techniques

### Markdown header split

Articles are split on `#` / `##` / `###` so a “Set up the Google tag” intro stays separate from “Track a button click” steps.

### Parent-child (small-to-big) chunks

| Piece | Size | Job |
|-------|------|-----|
| Child | ~500 characters, 80 overlap | What gets **embedded** and searched |
| Parent | up to ~2800 characters | What the **LLM actually reads** |

The child is stored as `heading + passage` so the vector is about “Track a button click”, not the page title. On search, extra children are fetched, then collapsed to unique parents (`similarity_search` in `indexer.py`).

This is a practical cousin of hierarchical RAG: search the specific sentence, return the whole how-to section.

```python
embed_text = f"{heading}\n\n{child}" if heading else child
# metadata keeps parent_text for the LLM
```

### Boilerplate filter

Google help pages often start with the same paragraph: “configure the Google tag… data to flow from your website…”. Indexing that intro made every query retrieve setup fluff. Chunks that look like that intro and have no procedure words (`click`, `trigger`, `event`, `dataLayer`, …) are dropped.

### Note dedupe

Before chunking, notes collapse:

- `http` vs `https` twins and trailing-slash twins
- identical article bodies (hash of text, ignoring the injected `Source:` line)

The longer official `https://…/answer/…` copy wins.

### Passage vs query embeddings

Nemotron `nvidia/nemotron-3-embed-1b:free` with `input_type=passage` at index time and `input_type=query` at search time. Chroma collection `martech_ga_gtm` uses cosine distance.

## Retrieval techniques (live path)

Order in `retrieve()`:

1. **Conversational follow-up merge** (rule, not an LLM). “What about SST?” after “track purchase” becomes `track purchase. Follow-up: What about SST?`.
2. **Query variants** — original question, `{core} Google Analytics 4`, `{core} Google Tag Manager`, plus a Cloud Run / tagging-server variant when the question is about server-side tagging.
3. **Multi-query dense retrieval** — embed each variant, search Chroma (`RETRIEVAL_TOP_K=8`), union, keep the best vector score per parent.
4. **Hybrid rescore** — `0.4 * vector + 0.6 * lexical overlap` (title/URL/headers weighted 0.65, body 0.35).
5. **Page-kind boost / penalty** — how-to pages get +0.4 when they match the question; API, legacy UA, ads, mobile, BigQuery get −0.5 unless the question asked for that kind. A how-to boost is skipped if the title barely overlaps the question.
6. **Diversity** — one chunk per source URL, then top `RERANK_TOP_N=5`.

```python
overlap = 0.65 * title_overlap + 0.35 * body_overlap
boost = source_adjustment(query, title, url, headers)
if title_overlap < 0.4 and boost > 0:
    boost = 0.0
chunk.score = 0.4 * chunk.score + 0.6 * overlap + boost
```

## Generation techniques

- Last **6 chat turns** go into the prompt (long assistant turns truncated).
- Each retrieved section gets **ready-to-paste quotes** (short sentences the model can cite).
- System prompt: one recommended method, GTM + GA4 steps, “copy this to your developer”. Vague asks get 2–3 clarifying questions first.

## Not on the live path

| Technique | Where it lives | Why it is unused in chat |
|-----------|----------------|--------------------------|
| LLM query rewrite | `rewrite_query` | Follow-up merge was more reliable |
| LLM query expansion | `expand_query` | Hand-built GA4/GTM variants were enough |
| LLM rerank | `rerank_chunks` | Slow; still surfaced ads/API pages |

GLM (`z-ai/glm-5.2`) is still used for **eval test-set generation** and **LLM-as-judge** (accuracy / completeness / relevance), not for live retrieve.

## Related files

- [README.md](../README.md) — how to run
- [tasks/plan.md](../tasks/plan.md) — why the design changed
- [specs/rag-poc.md](../specs/rag-poc.md) — success criteria
