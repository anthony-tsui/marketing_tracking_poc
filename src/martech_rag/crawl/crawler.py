"""Crawl4AI recursive crawler with GA/GTM content filter."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urldefrag, urljoin, urlparse, urlunparse

import html2text
import httpx
from bs4 import BeautifulSoup
from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    CacheMode,
    CrawlerRunConfig,
    UndetectedAdapter,
)
from crawl4ai.async_crawler_strategy import AsyncPlaywrightCrawlerStrategy

from martech_rag.config import Settings, get_settings

logger = logging.getLogger(__name__)

ALLOWED_HOSTS = {"developers.google.com", "support.google.com"}
CONTENT_FILTER = re.compile(
    r"google\s+analytics|google\s+tag\s+manager",
    re.IGNORECASE,
)
# Community / forum / social noise — never crawl these
BLOCKED_PATH_PARTS = (
    "/community",
    "/forum",
    "/forums",
    "/thread",
    "/threads",
    "/discussions",
    "/discussion",
    "/profile",
    "/users/",
    "/user/",
    "/comments",
    "/announcement",
    "/announcements",
    "/search",
    "/s/",
)
# Support help pages we keep
SUPPORT_ALLOWED_PREFIXES = (
    "/tagmanager/answer/",
    "/tagmanager/topic/",
    "/analytics/answer/",
    "/analytics/topic/",
)
SKIP_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".svg",
    ".webp",
    ".pdf",
    ".zip",
    ".css",
    ".js",
    ".ico",
    ".mp4",
    ".woff",
    ".woff2",
}
HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# Prefer these containers when converting HTML → markdown
CONTENT_SELECTORS = (
    ".article-content-container",
    ".article-container",
    "article.page",
    "article",
    ".devsite-article-body",
    "article.devsite-article",
    ".devsite-article",
    "#hcfe-content .main-content",
    "#hcfe-content",
    "[role='main']",
    "main",
)

NOISE_SELECTORS = (
    "script",
    "style",
    "noscript",
    "svg",
    "iframe",
    "nav",
    "header",
    "footer",
    "form",
    "figure",
    "video",
    "audio",
    "picture",
    "aside",
    "[role='navigation']",
    "[role='banner']",
    "[role='contentinfo']",
    "[role='complementary']",
    ".devsite-book-nav",
    ".devsite-footer",
    ".devsite-top-banner",
    ".devsite-header",
    ".devsite-toast",
    ".devsite-thumb-rating",
    ".skip-link",
    ".gc-feedback",
    ".feedback",
    ".related-articles",
    ".article-footer",
    ".youtube-player",
    ".video-container",
)

SEED_URLS = [
    "https://developers.google.com/analytics",
    "https://developers.google.com/tag-platform/tag-manager",
    "https://support.google.com/tagmanager/answer/14842164",
    "https://support.google.com/tagmanager/topic/14595647?hl=en",
    "https://support.google.com/tagmanager/topic/14598735?hl=en",
    "https://support.google.com/tagmanager/topic/14598641?hl=en",
    "https://support.google.com/tagmanager/topic/14226521?hl=en",
    "https://support.google.com/tagmanager/answer/14009343?hl=en",
    "https://support.google.com/tagmanager/topic/9001797?hl=en",
    "https://support.google.com/tagmanager/topic/9002095?hl=en",
    "https://support.google.com/tagmanager/topic/3281056?hl=en",
    "https://support.google.com/tagmanager/topic/7679384?hl=en",
    "https://support.google.com/tagmanager/topic/7683268?hl=en",
    "https://support.google.com/tagmanager/answer/9442095?hl=en",
]


@dataclass
class CrawledPage:
    url: str
    title: str
    markdown: str
    links: list[str] = field(default_factory=list)


@dataclass
class FetchResult:
    url: str
    success: bool
    title: str = ""
    markdown: str = ""
    links: list[str] = field(default_factory=list)
    error: str = ""


def normalize_url(url: str) -> str | None:
    raw = url.strip()
    if not raw or raw.startswith(("mailto:", "javascript:", "tel:", "#")):
        return None
    raw, _frag = urldefrag(raw)
    parsed = urlparse(raw)
    if parsed.scheme not in {"http", "https"}:
        return None
    host = parsed.netloc.lower()
    if host not in ALLOWED_HOSTS:
        return None
    path = parsed.path or "/"
    lower_path = path.lower()
    for ext in SKIP_EXTENSIONS:
        if lower_path.endswith(ext):
            return None
    query = parsed.query
    if "support.google.com" in host:
        parts = [p for p in query.split("&") if p.startswith("hl=")]
        query = "&".join(parts) if parts else "hl=en"
    else:
        query = ""
    normalized = urlunparse((parsed.scheme, host, path.rstrip("/") or "/", "", query, ""))
    return normalized


def path_allowed(url: str) -> bool:
    parsed = urlparse(url)
    path = parsed.path.lower().rstrip("/") or "/"

    # Never crawl community / forum / profile / search pages
    if any(part in path for part in BLOCKED_PATH_PARTS):
        return False
    if path.endswith("/community") or "/community/" in path + "/":
        return False

    host = parsed.netloc.lower()
    if host == "developers.google.com":
        return (
            path == "/analytics"
            or path.startswith("/analytics/")
            or path.startswith("/tag-platform/")
            or path.startswith("/tagmanager/")
        )
    if host == "support.google.com":
        # Only official answer/topic help articles — not community hubs
        return any(path.startswith(prefix) for prefix in SUPPORT_ALLOWED_PREFIXES)
    return False


def matches_content_filter(text: str) -> bool:
    return bool(CONTENT_FILTER.search(text or ""))


def _pick_content_root(soup: BeautifulSoup):
    for selector in CONTENT_SELECTORS:
        node = soup.select_one(selector)
        if node and len(node.get_text(" ", strip=True)) > 200:
            return node
    return soup.body or soup


def _strip_noise(root) -> None:
    for selector in NOISE_SELECTORS:
        for tag in root.select(selector):
            tag.decompose()
    # Keep link text only — remove href chrome from stored content
    for a in list(root.find_all("a")):
        text = a.get_text(" ", strip=True)
        lower = text.lower()
        href = (a.get("href") or "").lower()
        if lower in {
            "skip to main content",
            "sign in",
            "send feedback",
            "submit feedback",
            "help center",
            "community",
            "privacy policy",
            "terms of service",
            "next",
            "previous",
        }:
            a.decompose()
            continue
        if any(bad in href for bad in BLOCKED_PATH_PARTS):
            a.decompose()
            continue
        # Replace <a> with plain text so html2text won't emit markdown links
        a.replace_with(text if text else "")


def _clean_markdown(markdown: str) -> str:
    lines = []
    skip_prefixes = (
        "skip to main content",
        "sign in",
        "send feedback",
        "this help content",
        "general help center experience",
        "was this helpful",
        "need more help",
        "truefalse",
        "print this topic",
        "email this topic",
        "to view subtitles",
        "turn on youtube",
        "stay organized with collections",
        "save and categorize content",
        "outlined_flag",
        "next:",
        "previous:",
    )
    skip_exact = {
        "next",
        "previous",
        "home",
        "[",
        "]",
        "*",
        "* * *",
        "---",
        "outlined_flag",
        "send feedback",
    }
    for raw in markdown.splitlines():
        line = raw.strip()
        if not line:
            if lines and lines[-1] != "":
                lines.append("")
            continue
        lower = line.lower()
        if lower in skip_exact:
            continue
        if any(lower.startswith(p) for p in skip_prefixes):
            continue
        # Drop leftover url / markdown-link residue
        if lower.startswith("http://") or lower.startswith("https://") or lower.startswith("//"):
            continue
        if "](http" in lower or "](//" in lower or lower.startswith("[]("):
            continue
        if lower.startswith("[previous") or lower.startswith("[next"):
            continue
        # Drop "Page Summary" chrome often injected by developers.google.com
        if lower in {"page summary", "## page summary", "# page summary"}:
            continue
        lines.append(raw.rstrip())

    # Remove leading heading that is only "About X Stay organized..."
    cleaned_lines: list[str] = []
    for i, line in enumerate(lines):
        if i == 0 and "stay organized with collections" in line.lower():
            # Keep title text before that phrase if present
            cut = line.lower().find("stay organized with collections")
            title_part = line[:cut].strip(" #")
            if title_part:
                cleaned_lines.append(f"# {title_part}")
            continue
        cleaned_lines.append(line)

    cleaned: list[str] = []
    blank = 0
    for line in cleaned_lines:
        if not line.strip():
            blank += 1
            if blank <= 1:
                cleaned.append("")
            continue
        blank = 0
        cleaned.append(line)
    return "\n".join(cleaned).strip()


def html_to_markdown_and_links(url: str, html: str) -> FetchResult:
    soup = BeautifulSoup(html, "lxml")
    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()

    # Discover crawl links from full page; store text from main content only
    links: list[str] = []
    for a in soup.find_all("a", href=True):
        norm = normalize_url(urljoin(url, a["href"]))
        if norm and path_allowed(norm):
            links.append(norm)

    root = _pick_content_root(soup)
    _strip_noise(root)

    converter = html2text.HTML2Text()
    converter.ignore_links = True
    converter.ignore_images = True
    converter.ignore_tables = False
    converter.body_width = 0
    converter.single_line_break = False
    markdown = _clean_markdown(converter.handle(str(root)))

    return FetchResult(
        url=url,
        success=bool(markdown) and len(markdown) > 80,
        title=title or url,
        markdown=markdown,
        links=sorted(set(links)),
    )


class CrawlState:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.visited: set[str] = set()
        self.stored: set[str] = set()
        self.failed: set[str] = set()
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self.visited = set(data.get("visited", []))
        self.stored = set(data.get("stored", []))
        self.failed = set(data.get("failed", []))

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(
                {
                    "visited": sorted(self.visited),
                    "stored": sorted(self.stored),
                    "failed": sorted(self.failed),
                },
                indent=2,
            ),
            encoding="utf-8",
        )


def _extract_links_from_crawl4ai(result, base_url: str) -> list[str]:
    discovered: list[str] = []
    if not getattr(result, "links", None):
        return discovered
    internal = result.links.get("internal", []) or []
    external = result.links.get("external", []) or []
    for link in list(internal) + list(external):
        href = link.get("href") if isinstance(link, dict) else str(link)
        absolute = urljoin(base_url, href)
        norm = normalize_url(absolute)
        if norm and path_allowed(norm):
            discovered.append(norm)
    return discovered


async def fetch_httpx(client: httpx.AsyncClient, url: str) -> FetchResult:
    try:
        resp = await client.get(url, follow_redirects=True)
        resp.raise_for_status()
        return html_to_markdown_and_links(url, resp.text)
    except Exception as exc:  # noqa: BLE001
        logger.warning("HTTP fetch failed for %s: %s", url, exc)
        return FetchResult(url=url, success=False, error=str(exc))


async def crawl_docs(
    *,
    settings: Settings | None = None,
    seeds: list[str] | None = None,
    on_page=None,
    max_pages: int | None = None,
    reset_failed: bool = False,
) -> list[CrawledPage]:
    """BFS crawl from seeds; optionally call on_page(CrawledPage) for each kept page."""
    settings = settings or get_settings()
    seeds = seeds or SEED_URLS
    state = CrawlState(settings.crawl_state_path)
    if reset_failed:
        # Retry anything visited but never successfully stored (anti-bot empties, etc.)
        stale = (state.visited | state.failed) - state.stored
        state.visited -= stale
        state.failed.clear()
        state.save()

    queue: deque[str] = deque()
    for seed in seeds:
        n = normalize_url(seed)
        if n and path_allowed(n) and n not in state.visited:
            queue.append(n)

    kept: list[CrawledPage] = []
    browser_cfg = BrowserConfig(
        headless=True,
        verbose=False,
        enable_stealth=True,
        user_agent_mode="random",
    )
    run_cfg = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        word_count_threshold=10,
        exclude_external_links=False,
        process_iframes=False,
        remove_overlay_elements=True,
        magic=True,
        simulate_user=True,
        override_navigator=True,
        wait_until="domcontentloaded",
        page_timeout=45000,
        delay_before_return_html=0.8,
        remove_consent_popups=True,
    )

    sem = asyncio.Semaphore(settings.crawl_max_concurrency)
    crawler: AsyncWebCrawler | None = None
    if settings.crawl_use_browser:
        try:
            adapter = UndetectedAdapter()
            strategy = AsyncPlaywrightCrawlerStrategy(
                browser_config=browser_cfg,
                browser_adapter=adapter,
            )
            crawler = AsyncWebCrawler(crawler_strategy=strategy, config=browser_cfg)
            await crawler.start()
            logger.info("Crawl4AI browser started")
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Crawl4AI browser unavailable (%s). Continuing with HTTP-only crawl.",
                exc,
            )
            crawler = None
    else:
        logger.info("HTTP-only crawl (set CRAWL_USE_BROWSER=true to enable Playwright)")

    try:
        async with httpx.AsyncClient(headers=HTTP_HEADERS, timeout=60.0) as http:

            async def fetch(url: str) -> FetchResult:
                async with sem:
                    http_result = await fetch_httpx(http, url)
                    if http_result.success and len(http_result.markdown) > 200:
                        return http_result
                    if crawler is None or urlparse(url).netloc == "support.google.com":
                        return http_result

                    try:
                        result = await crawler.arun(url=url, config=run_cfg)
                    except Exception as crawl_exc:  # noqa: BLE001
                        logger.warning("Crawl4AI failed for %s: %s", url, crawl_exc)
                        return http_result

                    if result is None or not getattr(result, "success", False):
                        return http_result

                    # Prefer cleaned HTML through the same content extractor
                    cleaned_html = getattr(result, "cleaned_html", None) or ""
                    if cleaned_html and len(cleaned_html) > 200:
                        extracted = html_to_markdown_and_links(url, cleaned_html)
                        if extracted.success:
                            return extracted

                    markdown = _clean_markdown(
                        (result.markdown or cleaned_html or "") or ""
                    )
                    title = ""
                    if getattr(result, "metadata", None):
                        title = result.metadata.get("title") or ""
                    if not markdown.strip():
                        return http_result
                    return FetchResult(
                        url=url,
                        success=True,
                        title=title or url,
                        markdown=markdown.strip(),
                        links=_extract_links_from_crawl4ai(result, url)
                        or http_result.links,
                    )

            while queue:
                if max_pages is not None and len(kept) >= max_pages:
                    break

                batch: list[str] = []
                while queue and len(batch) < settings.crawl_max_concurrency:
                    if max_pages is not None and len(kept) >= max_pages:
                        break
                    url = queue.popleft()
                    if url in state.visited:
                        continue
                    state.visited.add(url)
                    batch.append(url)

                if not batch:
                    break

                results = await asyncio.gather(*(fetch(u) for u in batch))
                for result in results:
                    if max_pages is not None and len(kept) >= max_pages:
                        break
                    if not result.success:
                        state.failed.add(result.url)
                        continue

                    for norm in result.links:
                        if norm not in state.visited:
                            queue.append(norm)

                    if not matches_content_filter(result.markdown) and not matches_content_filter(
                        result.title
                    ):
                        continue

                    page = CrawledPage(
                        url=result.url,
                        title=(result.title or result.url).strip(),
                        markdown=result.markdown.strip(),
                        links=result.links,
                    )
                    if on_page is not None:
                        try:
                            await on_page(page)
                        except Exception as exc:  # noqa: BLE001
                            logger.warning(
                                "Notion store failed for %s: %s", result.url, exc
                            )
                            state.failed.add(result.url)
                            continue
                    kept.append(page)
                    state.stored.add(result.url)

                state.save()
                logger.info(
                    "Crawl progress: visited=%s kept=%s failed=%s queue=%s",
                    len(state.visited),
                    len(state.stored),
                    len(state.failed),
                    len(queue),
                )
    finally:
        if crawler is not None:
            try:
                await crawler.close()
            except Exception:  # noqa: BLE001
                pass

    state.save()
    return kept
