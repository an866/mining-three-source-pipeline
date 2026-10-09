"""OpenAI-compatible chat client.

One job: turn a list of messages into a string, or raise :class:`LLMError`. No
streaming, no tool calling, no retries-forever -- answer synthesis either succeeds
or degrades to the extractive answerer, and a retry loop in between would only make
a slow failure slower.

Used with DeepSeek (``LLM_BASE_URL=https://api.deepseek.com/v1``) by default, but
the request shape is the de-facto standard, so OpenAI, Together, Groq, vLLM and
Ollama's OpenAI shim all work by changing ``LLM_BASE_URL``.

The prompt is not built here. Keeping the client dumb is what makes it testable
with a stub transport and keeps the prompt -- the part that actually determines
answer quality -- in :mod:`pipeline.answer` where it is read by humans.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import httpx

from pipeline.config import LLMSettings, Settings, get_llm_settings, get_settings
from pipeline.errors import ConfigurationError, LLMError
from pipeline.logging_setup import get_logger

__all__ = ["ChatClient", "build_chat_client"]

_log = get_logger(__name__)


class ChatClient:
    """A minimal ``POST /chat/completions`` client."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        timeout: float = 60.0,
        settings: Settings | None = None,
    ) -> None:
        if not api_key:
            raise ConfigurationError("LLM_API_KEY is empty")
        if not base_url:
            raise ConfigurationError("LLM_BASE_URL is empty")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._settings = settings or get_settings()
        self._client = httpx.Client(
            timeout=httpx.Timeout(timeout),
            headers={"Authorization": f"Bearer {api_key}"},
        )

    def close(self) -> None:
        self._client.close()

    def complete(self, messages: Sequence[dict[str, str]], *, max_tokens: int | None = None) -> str:
        """Return the assistant's reply text.

        ``messages`` is the standard ``[{"role": ..., "content": ...}]`` list. The
        caller is responsible for the prompt; this method only transports it.
        """
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "temperature": self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
            "stream": False,
        }
        try:
            response = self._client.post(self._url, json=payload)
        except httpx.HTTPError as exc:
            raise LLMError(f"chat request failed: {exc}") from exc
        if response.status_code >= 400:
            raise LLMError(
                f"chat endpoint returned HTTP {response.status_code}: {response.text[:200]}"
            )
        try:
            body = response.json()
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise LLMError(f"unexpected chat response shape: {exc}") from exc
        if not isinstance(content, str) or not content.strip():
            raise LLMError("chat endpoint returned an empty message")
        return content.strip()


def build_chat_client(
    settings: Settings | None = None, llm_settings: LLMSettings | None = None
) -> ChatClient | None:
    """Build the client, or return ``None`` when no key is configured.

    ``None`` is a first-class outcome, not an error: the pipeline is designed to
    run end-to-end without a paid key, and the caller switches to the extractive
    answerer. That is what makes ``make demo`` work on a fresh clone.
    """
    resolved = settings or get_settings()
    llm = llm_settings or get_llm_settings()
    if not llm.chat_available:
        _log.info("no LLM key configured; answers will be extractive")
        return None
    return ChatClient(
        base_url=llm.base_url,
        api_key=llm.api_key,
        model=llm.model,
        temperature=llm.temperature,
        max_tokens=llm.max_tokens,
        settings=resolved,
    )
