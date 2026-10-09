"""Logging configuration.

MCP servers speak JSON-RPC on **stdout**, so logging must never write there.
Every handler in this module targets ``stderr``. That single constraint is the
reason this module exists rather than a bare ``logging.basicConfig`` call.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any, Final

from pipeline.config import Settings, get_settings

__all__ = ["configure_logging", "get_logger"]

_LEVEL_COLORS: Final[dict[str, str]] = {
    "DEBUG": "\033[36m",
    "INFO": "\033[32m",
    "WARNING": "\033[33m",
    "ERROR": "\033[31m",
    "CRITICAL": "\033[35m",
}
_RESET: Final[str] = "\033[0m"
_CONFIGURED: set[str] = set()


class _JsonFormatter(logging.Formatter):
    """Minimal, dependency-free structured formatter."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        for key, value in getattr(record, "extra_fields", {}).items():
            payload[key] = value
        return json.dumps(payload, ensure_ascii=False, default=str)


class _PrettyFormatter(logging.Formatter):
    """Human-readable single-line formatter with optional colour."""

    def __init__(self) -> None:
        super().__init__("%(asctime)s %(levelname)-8s %(name)-28s %(message)s", "%H:%M:%S")
        self._use_color = sys.stderr.isatty()

    def format(self, record: logging.LogRecord) -> str:
        rendered = super().format(record)
        extra = getattr(record, "extra_fields", None)
        if extra:
            rendered = f"{rendered} {json.dumps(extra, ensure_ascii=False, default=str)}"
        if self._use_color:
            color = _LEVEL_COLORS.get(record.levelname, "")
            rendered = f"{color}{rendered}{_RESET}"
        return rendered


def configure_logging(settings: Settings | None = None, *, force: bool = False) -> None:
    """Install a stderr handler on the root logger.

    Idempotent: repeated calls are no-ops unless ``force=True``.
    """
    resolved = settings or get_settings()
    signature = f"{resolved.log_level}:{resolved.log_json}"
    if not force and signature in _CONFIGURED:
        return
    _CONFIGURED.add(signature)

    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(_JsonFormatter() if resolved.log_json else _PrettyFormatter())

    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(resolved.log_level)

    # These libraries are chatty at DEBUG and drown out our own records.
    for noisy in ("httpx", "httpcore", "urllib3", "openai"):
        logging.getLogger(noisy).setLevel(max(logging.WARNING, root.level))


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger, configuring logging on first use."""
    configure_logging()
    return logging.getLogger(name)
