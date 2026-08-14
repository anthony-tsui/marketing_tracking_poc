"""Read project Web Clip notes from Notion."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from martech_rag.config import Settings, get_settings
from martech_rag.notion_api import make_async_client, notion_call

logger = logging.getLogger(__name__)


@dataclass
class NotionNote:
    page_id: str
    title: str
    url: str
    markdown: str


def _rich_text_to_str(rich: list[dict[str, Any]] | None) -> str:
    if not rich:
        return ""
    return "".join(part.get("plain_text", "") for part in rich)


class NotionReader:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.client = make_async_client(self.settings)

    async def aclose(self) -> None:
        await self.client.aclose()

    async def list_project_notes(self) -> list[NotionNote]:
        notes: list[NotionNote] = []
        cursor = None
        while True:
            kwargs: dict[str, Any] = {
                "data_source_id": self.settings.notion_notes_data_source_id,
                "filter": {
                    "and": [
                        {
                            "property": "Project",
                            "relation": {"contains": self.settings.notion_project_id},
                        },
                        {
                            "property": "Type",
                            "select": {"equals": "Web Clip"},
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
                what="list project web clips",
            )
            for page in resp.get("results") or []:
                props = page.get("properties") or {}
                title = _rich_text_to_str((props.get("Name") or {}).get("title"))
                url = (props.get("URL") or {}).get("url") or ""
                page_id = page["id"]
                try:
                    md_resp = await notion_call(
                        lambda pid=page_id: self.client.pages.retrieve_markdown(
                            page_id=pid
                        ),
                        what=f"retrieve_markdown {page_id}",
                    )
                    markdown = md_resp.get("markdown") or ""
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Skip markdown for %s: %s", page_id, exc)
                    markdown = ""
                notes.append(
                    NotionNote(
                        page_id=page_id,
                        title=title or url or page_id,
                        url=url,
                        markdown=markdown,
                    )
                )
            if not resp.get("has_more"):
                break
            cursor = resp.get("next_cursor")
        logger.info("Loaded %s Notion Web Clips for project", len(notes))
        return notes
