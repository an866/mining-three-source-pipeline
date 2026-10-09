"""HTTP access with retries, timeouts and an offline guard.

Rules encoded here:

* Every outbound request goes through :class:`HttpClient`, so the offline flag
  is enforced in exactly one place.
* Transient failures (connection errors, 429, 5xx) are retried with exponential
  backoff; 4xx other than 429 fails fast because retrying will not help.
* On final failure we raise :class:`SourceUnavailableError`, which the callers treat
  as "degrade to cache/fixture", never as a crash.
"""

from __future__ import annotations

import json
from types import TracebackType
from typing import Any, Self

import httpx
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from pipeline.config import Settings, get_settings
from pipeline.errors import OfflineModeError, ParseError, SourceUnavailableError
from pipeline.logging_setup import get_logger

__all__ = ["HttpClient", "TransientHttpError"]

_log = get_logger(__name__)

_RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504, 522, 524})


class TransientHttpError(Exception):
    """Internal marker for a failure worth retrying."""


def _log_retry(state: RetryCallState) -> None:
    exc = state.outcome.exception() if state.outcome else None
    _log.debug(
        "retrying http request",
        extra={"extra_fields": {"attempt": state.attempt_number, "error": str(exc)}},
    )


class HttpClient:
    """A thin, reusable wrapper around :class:`httpx.Client`.

    Usable as a context manager::

        with HttpClient(source="mining.com") as http:
            payload = http.get_json("https://www.mining.com/feed/")
    """

    def __init__(
        self,
        *,
        source: str,
        settings: Settings | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.source = source
        self._settings = settings or get_settings()
        base_headers = {
            "User-Agent": self._settings.user_agent,
            "Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.8",
        }
        if headers:
            base_headers.update(headers)
        self._client = httpx.Client(
            timeout=httpx.Timeout(self._settings.http_timeout),
            follow_redirects=True,
            headers=base_headers,
            proxy=self._settings.http_proxy or None,
        )

    # -- lifecycle -----------------------------------------------------------

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # -- verbs ---------------------------------------------------------------

    def get_bytes(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> bytes:
        """GET ``url`` and return the raw body."""
        return self._request(url, params=params, headers=headers).content

    def get_text(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> str:
        """GET ``url`` and decode the body as text."""
        response = self._request(url, params=params, headers=headers)
        return response.text

    def get_json(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> Any:
        """GET ``url`` and decode the body as JSON."""
        text = self.get_text(url, params=params, headers=headers)
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise ParseError(f"JSON from {url}", str(exc)) from exc

    # -- internals -----------------------------------------------------------

    def _guard_online(self) -> None:
        if self._settings.offline:
            raise OfflineModeError(self.source)

    def _request(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        # Single choke point for the offline policy: every verb funnels through
        # here, so the guard cannot be forgotten in one code path.
        self._guard_online()

        @retry(
            reraise=True,
            stop=stop_after_attempt(self._settings.http_retries),
            wait=wait_exponential(multiplier=0.6, max=8),
            retry=retry_if_exception_type((TransientHttpError, httpx.TransportError)),
            before_sleep=_log_retry,
        )
        def _attempt() -> httpx.Response:
            try:
                response = self._client.get(url, params=params, headers=headers)
            except httpx.TransportError as exc:
                raise TransientHttpError(str(exc)) from exc
            if response.status_code in _RETRYABLE_STATUS:
                raise TransientHttpError(f"HTTP {response.status_code}")
            return response

        try:
            response = _attempt()
        except (TransientHttpError, httpx.TransportError) as exc:
            raise SourceUnavailableError(self.source, f"{url} -> {exc}") from exc

        if response.status_code >= 400:
            # 4xx (non-retryable): the source is reachable but rejecting us,
            # e.g. a login wall or a bot filter.
            raise SourceUnavailableError(
                self.source,
                f"{url} -> HTTP {response.status_code}",
            )
        return response
