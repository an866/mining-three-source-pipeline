"""Error taxonomy.

Every failure raised by this project derives from :class:`PipelineError`, so
callers can tell "the upstream source is down" (recoverable -- degrade to the
cache or a bundled fixture) from "we parsed garbage" (a real bug worth
surfacing) without resorting to ``except Exception``.
"""

from __future__ import annotations

__all__ = [
    "ConfigurationError",
    "LLMError",
    "OfflineModeError",
    "ParseError",
    "PipelineError",
    "SourceUnavailableError",
    "VectorStoreError",
]


class PipelineError(Exception):
    """Base class for every error raised by this project."""


class ConfigurationError(PipelineError):
    """Raised when required configuration is missing or contradictory."""


class SourceUnavailableError(PipelineError):
    """An upstream source could not be reached, or answered with an error.

    Recoverable: callers should fall back to a cached response or a bundled
    fixture rather than aborting the run.
    """

    def __init__(self, source: str, detail: str) -> None:
        super().__init__(f"source={source!r} unavailable: {detail}")
        self.source = source
        self.detail = detail


class OfflineModeError(SourceUnavailableError):
    """Raised when a network call is attempted while ``MINING_OFFLINE=1``."""

    def __init__(self, source: str) -> None:
        super().__init__(source, "offline mode is enabled; network access is forbidden")


class ParseError(PipelineError):
    """A payload was retrieved but could not be understood."""

    def __init__(self, what: str, detail: str) -> None:
        super().__init__(f"failed to parse {what}: {detail}")
        self.what = what
        self.detail = detail


class LLMError(PipelineError):
    """The configured language model failed or returned an unusable response."""


class VectorStoreError(PipelineError):
    """The vector index is missing, corrupt, or dimensionally inconsistent."""

    def __init__(self, detail: str) -> None:
        super().__init__(f"vector store error: {detail}")
        self.detail = detail
