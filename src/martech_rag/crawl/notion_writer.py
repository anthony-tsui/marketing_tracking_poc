"""Write crawled pages into Notion Notes as Web Clips."""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any
from uuid import UUID

from notion_client.errors import APIResponseError, RequestTimeoutError

from martech_rag.config import Settings, get_settings
from martech_rag.crawl.crawler import CrawledPage
from martech_rag.notion_api import make_async_client, notion_call

logger = logging.getLogger(__name__)

_MAX_MARKDOWN_CHARS = 40_000


def _notion_id(value: str) -> str:
    """Normalize Notion IDs to dashed UUID form."""
    raw = value.strip().replace("-", "")
    if len(raw) != 32 or not re.fullmatch(r"[0-9a-fA-F]+", raw):
        return value
    return str(UUID(raw))


class NotionWriter:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.client = make_async_client(self.settings)
        self.project_id = _notion_id(self.settings.notion_project_id)
        self.tag_id = _notion_id(self.settings.notion_martech_tag_id)
        self._prop_lock = asyncio.Lock()

    async def aclose(self) -> None:
        await self.client.aclose()

    async def find_by_url(self, url: str) -> str | None:
        resp = await notion_call(
            lambda: self.client.data_sources.query(
                data_source_id=self.settings.notion_notes_data_source_id,
                filter={"property": "URL", "url": {"equals": url}},
                page_size=1,
            ),
            what=f"find_by_url {url}",
        )
        results = resp.get("results") or []
        if not results:
            return None
        return results[0]["id"]

    def _properties(self, page: CrawledPage) -> dict[str, Any]:
        title = page.title[:2000] if page.title else page.url
        return {
            "Name": {"title": [{"type": "text", "text": {"content": title}}]},
            "Type": {"select": {"name": "Web Clip"}},
            "URL": {"url": page.url},
            "Archived": {"checkbox": False},
            "Project": {"relation": [{"id": self.project_id}]},
            "Tag": {"relation": [{"id": self.tag_id}]},
        }

    def _label_properties(self) -> dict[str, Any]:
        """Core labels that must always be present on crawled notes."""
        return {
            "Type": {"select": {"name": "Web Clip"}},
            "Archived": {"checkbox": False},
            "Project": {"relation": [{"id": self.project_id}]},
            "Tag": {"relation": [{"id": self.tag_id}]},
        }

    @staticmethod
    def _body_markdown(page: CrawledPage) -> str:
        header = f"# {page.title}\n\nSource: {page.url}\n\n"
        body = page.markdown.strip()
        content = header + body
        if len(content) > _MAX_MARKDOWN_CHARS:
            content = (
                content[:_MAX_MARKDOWN_CHARS]
                + "\n\n...(truncated; see source URL for full document)\n"
            )
        return content

    async def _set_markdown(self, page_id: str, markdown: str) -> None:
        await notion_call(
            lambda: self.client.pages.update_markdown(
                page_id=page_id,
                type="replace_content",
                replace_content={"new_str": markdown},
                allow_async=True,
            ),
            what=f"update_markdown {page_id}",
        )

    @staticmethod
    def _relation_ids(props: dict[str, Any], name: str) -> set[str]:
        items = ((props.get(name) or {}).get("relation")) or []
        return {_notion_id(item.get("id", "")) for item in items if item.get("id")}

    async def _labels_ok(self, page_id: str) -> bool:
        page = await notion_call(
            lambda: self.client.pages.retrieve(page_id=page_id),
            what=f"retrieve {page_id}",
        )
        props = page.get("properties") or {}
        typ = ((props.get("Type") or {}).get("select") or {}).get("name")
        projects = self._relation_ids(props, "Project")
        tags = self._relation_ids(props, "Tag")
        return (
            typ == "Web Clip"
            and self.project_id in projects
            and self.tag_id in tags
            and not page.get("archived", False)
        )

    async def _ensure_labels(self, page_id: str, *, url: str) -> None:
        """Verify Project/Tag/Type and repair if Notion dropped them."""
        for attempt in range(1, 4):
            try:
                if await self._labels_ok(page_id):
                    return
                logger.warning(
                    "Missing Project/Tag/Type on %s (attempt %s) — repairing",
                    url,
                    attempt,
                )
                await notion_call(
                    lambda: self.client.pages.update(
                        page_id=page_id,
                        archived=False,
                        properties=self._label_properties(),
                    ),
                    what=f"repair labels {page_id}",
                )
                await asyncio.sleep(0.4 * attempt)
            except (APIResponseError, RequestTimeoutError) as exc:
                logger.warning(
                    "Label repair failed for %s attempt %s: %s",
                    url,
                    attempt,
                    exc,
                )
                await asyncio.sleep(0.5 * attempt)
        if not await self._labels_ok(page_id):
            logger.error(
                "Could not set Project/Tag on Notion note for %s (%s)",
                url,
                page_id,
            )

    async def upsert_page(self, page: CrawledPage) -> str:
        props = self._properties(page)
        markdown = self._body_markdown(page)
        # Serialize property writes lightly to reduce intermittent relation drops
        async with self._prop_lock:
            existing_id = await self.find_by_url(page.url)
            if existing_id:
                await notion_call(
                    lambda: self.client.pages.update(
                        page_id=existing_id,
                        archived=False,
                        properties=props,
                    ),
                    what=f"update {page.url}",
                )
                page_id = existing_id
                logger.info("Updated Notion note for %s", page.url)
            else:
                # Create properties first (no async markdown) so relations stick
                created = await notion_call(
                    lambda: self.client.pages.create(
                        parent={"database_id": self.settings.notion_notes_database_id},
                        properties=props,
                    ),
                    what=f"create {page.url}",
                )
                page_id = created["id"]
                logger.info("Created Notion note for %s", page.url)

        try:
            await self._set_markdown(page_id, markdown)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Markdown write failed for %s: %s", page.url, exc)

        await self._ensure_labels(page_id, url=page.url)
        await asyncio.sleep(0.25)
        return page_id

    async def repair_missing_labels(self) -> dict[str, int]:
        """Scan project Web Clips and repair missing Project/Tag labels."""
        fixed = 0
        checked = 0
        cursor = None
        while True:
            kwargs: dict[str, Any] = {
                "data_source_id": self.settings.notion_notes_data_source_id,
                "filter": {
                    "and": [
                        {
                            "property": "Type",
                            "select": {"equals": "Web Clip"},
                        },
                        {
                            "property": "URL",
                            "url": {"is_not_empty": True},
                        },
                        {
                            "property": "Archived",
                            "checkbox": {"equals": False},
                        },
                    ]
                },
                "page_size": 50,
            }
            if cursor:
                kwargs["start_cursor"] = cursor
            resp = await notion_call(
                lambda: self.client.data_sources.query(**kwargs),
                what="list web clips for label repair",
            )
            for page in resp.get("results") or []:
                checked += 1
                page_id = page["id"]
                props = page.get("properties") or {}
                url = (props.get("URL") or {}).get("url") or page_id
                projects = self._relation_ids(props, "Project")
                tags = self._relation_ids(props, "Tag")
                needs = self.project_id not in projects or self.tag_id not in tags
                if needs:
                    await self._ensure_labels(page_id, url=url)
                    fixed += 1
            if not resp.get("has_more"):
                break
            cursor = resp.get("next_cursor")
        return {"checked": checked, "fixed": fixed}
