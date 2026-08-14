"""Shared Notion client with timeout + retry handling."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import TypeVar

import httpx
from notion_client import AsyncClient, RetryOptions
from notion_client.errors import APIErrorCode, APIResponseError, RequestTimeoutError
from tenacity import (
    AsyncRetrying,
    before_sleep_log,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from martech_rag.config import Settings, get_settings

logger = logging.getLogger(__name__)

T = TypeVar("T")

_TRANSIENT_API_CODES = {
    APIErrorCode.RateLimited,
    APIErrorCode.ServiceUnavailable,
    APIErrorCode.GatewayTimeout,
    APIErrorCode.InternalServerError,
    APIErrorCode.ConflictError,
}


def is_transient_notion_error(exc: BaseException) -> bool:
    """True for timeouts, rate limits, and other retryable Notion failures."""
    if isinstance(exc, RequestTimeoutError):
        return True
    if isinstance(exc, (httpx.TimeoutException, httpx.TransportError)):
        return True
    if APIResponseError.is_api_response_error(exc):
        return exc.code in _TRANSIENT_API_CODES
    return False


def make_async_client(settings: Settings | None = None) -> AsyncClient:
    settings = settings or get_settings()
    return AsyncClient(
        auth=settings.notion_token,
        timeout_ms=settings.notion_timeout_ms,
        retry=RetryOptions(
            max_retries=settings.notion_max_retries,
            initial_retry_delay_ms=2_000,
            max_retry_delay_ms=60_000,
        ),
    )


async def notion_call(op: Callable[[], Awaitable[T]], *, what: str) -> T:
    """Run a Notion request, retrying client-side timeouts the SDK does not retry."""
    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(5),
        wait=wait_exponential_jitter(initial=2, max=30),
        retry=retry_if_exception(is_transient_notion_error),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    ):
        with attempt:
            try:
                return await op()
            except Exception as exc:
                if is_transient_notion_error(exc):
                    logger.warning(
                        "Transient Notion error on %s (attempt %s): %s",
                        what,
                        attempt.retry_state.attempt_number,
                        exc,
                    )
                raise
    raise RuntimeError(f"Notion call failed: {what}")  # pragma: no cover
