"""Build / refresh Chroma index from Notion notes."""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from martech_rag.config import Settings, get_settings
from martech_rag.ingest.chunking import chunk_notion_notes, dedupe_notes
from martech_rag.ingest.notion_reader import NotionReader
from martech_rag.llm.openrouter import OpenRouterClient

logger = logging.getLogger(__name__)


class OpenRouterEmbeddings(Embeddings):
    """LangChain Embeddings adapter over async OpenRouter client (sync wrappers)."""

    def __init__(self, client: OpenRouterClient, input_type: str = "passage") -> None:
        self.client = client
        self.input_type = input_type

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        import asyncio

        return asyncio.run(
            self.client.embed_batched(texts, input_type=self.input_type)
        )

    def embed_query(self, text: str) -> list[float]:
        import asyncio

        vectors = asyncio.run(self.client.embed([text], input_type="query"))
        return vectors[0]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self.client.embed_batched(texts, input_type=self.input_type)

    async def aembed_query(self, text: str) -> list[float]:
        vectors = await self.client.embed([text], input_type="query")
        return vectors[0]


def get_chroma_collection(settings: Settings | None = None):
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    settings = settings or get_settings()
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path=str(settings.chroma_dir),
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    return client.get_or_create_collection(
        name="martech_ga_gtm",
        metadata={"hnsw:space": "cosine"},
    )


async def index_from_notion(
    settings: Settings | None = None,
    *,
    reset: bool = True,
) -> int:
    settings = settings or get_settings()
    reader = NotionReader(settings)
    llm = OpenRouterClient(settings)
    try:
        notes = await reader.list_project_notes()
        notes = dedupe_notes(notes)
        docs = chunk_notion_notes(notes, settings)
        if not docs:
            logger.warning("No documents to index")
            return 0

        # Dedupe by chunk_id (safety net for identical section pieces)
        unique_docs: list[Document] = []
        seen_ids: set[str] = set()
        for doc in docs:
            cid = str(doc.metadata.get("chunk_id") or "")
            if not cid or cid in seen_ids:
                continue
            seen_ids.add(cid)
            unique_docs.append(doc)
        if len(unique_docs) < len(docs):
            logger.info(
                "Dropped %s duplicate chunk ids before upsert",
                len(docs) - len(unique_docs),
            )
        docs = unique_docs

        collection = get_chroma_collection(settings)
        if reset:
            # Drop and recreate for clean rebuild
            import chromadb
            from chromadb.config import Settings as ChromaSettings

            client = chromadb.PersistentClient(
                path=str(settings.chroma_dir),
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            try:
                client.delete_collection("martech_ga_gtm")
            except Exception:  # noqa: BLE001
                pass
            collection = client.get_or_create_collection(
                name="martech_ga_gtm",
                metadata={"hnsw:space": "cosine"},
            )

        ids = [d.metadata["chunk_id"] for d in docs]
        texts = [d.page_content for d in docs]
        metadatas: list[dict[str, Any]] = [
            {
                "chunk_id": d.metadata.get("chunk_id", ""),
                "parent_id": d.metadata.get("parent_id", ""),
                "parent_text": d.metadata.get("parent_text", ""),
                "notion_page_id": d.metadata.get("notion_page_id", ""),
                "source_url": d.metadata.get("source_url", ""),
                "title": d.metadata.get("title", ""),
                "headers": d.metadata.get("headers", ""),
            }
            for d in docs
        ]

        embeddings = await llm.embed_batched(texts, input_type="passage")
        batch = settings.embed_batch_size
        for i in range(0, len(docs), batch):
            collection.upsert(
                ids=ids[i : i + batch],
                documents=texts[i : i + batch],
                metadatas=metadatas[i : i + batch],
                embeddings=embeddings[i : i + batch],
            )
        logger.info("Indexed %s chunks into Chroma", len(docs))
        return len(docs)
    finally:
        await reader.aclose()
        await llm.aclose()


async def similarity_search(
    query: str,
    *,
    k: int = 8,
    settings: Settings | None = None,
    client: OpenRouterClient | None = None,
) -> list[Document]:
    settings = settings or get_settings()
    owns_client = client is None
    client = client or OpenRouterClient(settings)
    try:
        query_vec = (await client.embed([query], input_type="query"))[0]
        collection = get_chroma_collection(settings)
        # Fetch extra children, then collapse to unique parent sections
        result = collection.query(
            query_embeddings=[query_vec],
            n_results=max(k * 3, 12),
            include=["documents", "metadatas", "distances"],
        )
        docs: list[Document] = []
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        seen_parents: set[str] = set()
        for content, meta, dist in zip(documents, metadatas, distances):
            meta = dict(meta or {})
            parent_id = str(meta.get("parent_id") or meta.get("chunk_id") or "")
            if parent_id and parent_id in seen_parents:
                continue
            if parent_id:
                seen_parents.add(parent_id)
            score = 1.0 - float(dist) if dist is not None else 0.0
            meta["score"] = score
            body = (meta.get("parent_text") or "").strip() or (content or "")
            docs.append(Document(page_content=body, metadata=meta))
            if len(docs) >= k:
                break
        return docs
    finally:
        if owns_client:
            await client.aclose()
