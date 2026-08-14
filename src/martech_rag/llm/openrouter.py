"""Async OpenRouter client for chat completions and embeddings."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential

from martech_rag.config import Settings, get_settings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


def _message_text(message: dict[str, Any] | None, *, allow_reasoning: bool = False) -> str:
    """Prefer visible content. Reasoning models may stash the answer in other fields."""
    if not message:
        return ""
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                parts.append(str(part.get("text") or part.get("content") or ""))
        joined = "".join(parts).strip()
        if joined:
            return joined
    if not allow_reasoning:
        return ""
    for key in ("reasoning", "reasoning_content"):
        value = message.get(key)
        if isinstance(value, str) and value.strip():
            return value
    details = message.get("reasoning_details") or []
    texts: list[str] = []
    for item in details:
        if isinstance(item, dict):
            text = item.get("text") or item.get("content") or ""
            if text:
                texts.append(str(text))
    return "\n".join(texts).strip()


_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.S | re.I)
_ANSWER_HEAD = re.compile(
    r"^(\*\*)?(In short|What this means for you|What to do in GTM|What to do in GA4)(\*\*)?:?",
    re.M | re.I,
)
_REASONING_LEAK = re.compile(
    r"^(The user asks|Let me |I think |Hmm\.|Should I |Wait —|I'll |I should )",
    re.M,
)


def strip_reasoning(text: str) -> str:
    """Drop hidden thinking if a reasoning model leaked it into the visible answer."""
    cleaned = _THINK_BLOCK.sub("", text or "").strip()
    leaked = bool(
        _REASONING_LEAK.search(cleaned[:800]) or cleaned.startswith("The user asks")
    )
    if not leaked:
        return cleaned
    matches = list(_ANSWER_HEAD.finditer(cleaned))
    for match in reversed(matches):
        rest = cleaned[match.start() :].strip()
        if len(rest) > 200:
            return rest
    return cleaned


def _strip_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end > start:
        return cleaned[start : end + 1]
    return cleaned


class OpenRouterClient:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._client = httpx.AsyncClient(
            base_url=self.settings.openrouter_base_url,
            headers={
                "Authorization": f"Bearer {self.settings.openrouter_api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/martech-rag-poc",
                "X-Title": "Marketing Tracking RAG PoC",
            },
            timeout=httpx.Timeout(120.0, connect=30.0),
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> OpenRouterClient:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.aclose()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=20))
    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> str:
        payload: dict[str, Any] = {
            "model": model or self.settings.chat_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "reasoning": {"enabled": False, "exclude": True},
        }
        resp = await self._client.post("/chat/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        text = strip_reasoning(_message_text(message, allow_reasoning=False))
        if not text:
            logger.error(
                "Empty chat content finish_reason=%s message_keys=%s",
                choice.get("finish_reason"),
                list(message.keys()),
            )
            raise ValueError("OpenRouter returned an empty chat message")
        if choice.get("finish_reason") == "length" and len(text) < 12000:
            logger.warning("Chat hit max_tokens; requesting continuation")
            cont_messages = [
                *messages,
                {"role": "assistant", "content": text},
                {
                    "role": "user",
                    "content": (
                        "Continue the answer from exactly where you stopped. "
                        "Do not repeat earlier sections. Finish GTM, GA4, and "
                        "Copy this to your developer."
                    ),
                },
            ]
            cont_payload = {**payload, "messages": cont_messages}
            cont_resp = await self._client.post("/chat/completions", json=cont_payload)
            cont_resp.raise_for_status()
            cont_choice = (cont_resp.json().get("choices") or [{}])[0]
            cont_text = strip_reasoning(
                _message_text(cont_choice.get("message") or {}, allow_reasoning=False)
            )
            if cont_text:
                text = f"{text.rstrip()}\n\n{cont_text}"
        return text

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=20), reraise=True)
    async def chat_structured(
        self,
        messages: list[dict[str, str]],
        schema: type[T],
        *,
        model: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ) -> T:
        """Call chat with JSON schema response format and parse into Pydantic."""
        schema_json = schema.model_json_schema()
        payload: dict[str, Any] = {
            "model": model or self.settings.retrieval_model,
            "messages": [
                *messages,
                {
                    "role": "system",
                    "content": (
                        "Respond with valid JSON only that matches the required schema. "
                        f"Schema: {json.dumps(schema_json)}"
                    ),
                },
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
            # GLM-5.2 puts JSON in `reasoning` unless thinking is turned off
            "reasoning": {"enabled": False},
        }
        resp = await self._client.post("/chat/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        content = _message_text(message, allow_reasoning=True)
        if not content:
            logger.error(
                "Empty structured content finish_reason=%s keys=%s usage=%s",
                choice.get("finish_reason"),
                list(message.keys()),
                data.get("usage"),
            )
            raise ValueError(f"OpenRouter returned empty JSON for {schema.__name__}")
        cleaned = _strip_fences(content)
        # Some models wrap the object under a single top-level key
        try:
            return schema.model_validate_json(cleaned)
        except Exception:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict) and len(parsed) == 1:
                only = next(iter(parsed.values()))
                if isinstance(only, dict):
                    return schema.model_validate(only)
            logger.error("Structured parse failed for %s: %s", schema.__name__, cleaned[:500])
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=20))
    async def embed(
        self,
        texts: list[str],
        *,
        input_type: str = "passage",
        model: str | None = None,
    ) -> list[list[float]]:
        if not texts:
            return []
        payload: dict[str, Any] = {
            "model": model or self.settings.embed_model,
            "input": texts if len(texts) > 1 else texts[0],
            "input_type": input_type,
        }
        resp = await self._client.post("/embeddings", json=payload)
        resp.raise_for_status()
        data = resp.json()
        items = sorted(data["data"], key=lambda x: x["index"])
        return [item["embedding"] for item in items]

    async def embed_batched(
        self,
        texts: list[str],
        *,
        input_type: str = "passage",
        batch_size: int | None = None,
    ) -> list[list[float]]:
        size = batch_size or self.settings.embed_batch_size
        vectors: list[list[float]] = []
        for i in range(0, len(texts), size):
            batch = texts[i : i + size]
            vectors.extend(await self.embed(batch, input_type=input_type))
        return vectors
