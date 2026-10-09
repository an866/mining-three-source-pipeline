"""On-disk response cache and bundled-fixture loader.

Two jobs, one module, because they are two halves of the same policy:

1. **Cache** -- a network response is stored under ``<cache_dir>/<namespace>/``
   keyed by a hash of the request, with a TTL. A stale entry is still returned
   when the upstream is unreachable (``stale-while-error``), which is what keeps
   the pipeline alive during an outage.
2. **Fixtures** -- immutable sample payloads shipped inside each package so the
   project is fully reproducible with ``MINING_OFFLINE=1``.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar, cast

from pipeline.config import Settings, get_settings
from pipeline.logging_setup import get_logger

__all__ = ["DiskCache"]

_log = get_logger(__name__)
T = TypeVar("T")


def _hash_key(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]


class DiskCache:
    """A tiny, dependency-free JSON cache with TTL and a stale-while-error mode."""

    def __init__(
        self,
        root: Path | None = None,
        ttl: int | None = None,
        *,
        enabled: bool | None = None,
        settings: Settings | None = None,
    ) -> None:
        resolved = settings or get_settings()
        self.root = Path(root) if root is not None else resolved.cache_dir
        self.ttl = resolved.cache_ttl if ttl is None else ttl
        self.enabled = resolved.cache_enabled if enabled is None else enabled

    # -- paths ---------------------------------------------------------------

    def _entry_path(self, namespace: str, key: str) -> Path:
        return self.root / namespace / f"{_hash_key(key)}.json"

    # -- basic get/set -------------------------------------------------------

    def get(self, namespace: str, key: str) -> Any | None:
        """Return the cached value, or ``None`` when absent/expired/disabled."""
        if not self.enabled:
            return None
        path = self._entry_path(namespace, key)
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:  # pragma: no cover - corrupt entry
            _log.warning(
                "cache entry unreadable, ignoring",
                extra={"extra_fields": {"path": str(path), "error": str(exc)}},
            )
            return None
        if self._is_expired(payload):
            return None
        return payload.get("value")

    def get_stale(self, namespace: str, key: str) -> Any | None:
        """Return a cached value regardless of TTL (used when upstream fails)."""
        if not self.enabled:
            return None
        path = self._entry_path(namespace, key)
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8")).get("value")
        except (json.JSONDecodeError, OSError):
            return None

    def set(self, namespace: str, key: str, value: Any, ttl: int | None = None) -> Path | None:
        """Persist ``value``; returns the written path (or ``None`` when disabled)."""
        if not self.enabled:
            return None
        path = self._entry_path(namespace, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "key": key,
            "namespace": namespace,
            "cached_at": time.time(),
            "ttl": self.ttl if ttl is None else ttl,
            "value": value,
        }
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, default=str), encoding="utf-8")
        tmp.replace(path)
        return path

    def get_or_set(
        self,
        namespace: str,
        key: str,
        producer: Callable[[], T],
        ttl: int | None = None,
    ) -> T:
        """Return a cached value, otherwise compute, store and return it."""
        cached = self.get(namespace, key)
        if cached is not None:
            return cast(T, cached)
        value = producer()
        self.set(namespace, key, value, ttl=ttl)
        return value

    # -- maintenance ---------------------------------------------------------

    def _is_expired(self, payload: dict[str, Any]) -> bool:
        ttl = payload.get("ttl", self.ttl)
        if ttl in (None, 0):
            return False
        return (time.time() - float(payload.get("cached_at", 0))) > float(ttl)

    def clear(self, namespace: str | None = None) -> int:
        """Delete cached entries; returns the number of files removed."""
        target = self.root / namespace if namespace else self.root
        if not target.exists():
            return 0
        removed = 0
        for path in target.rglob("*.json"):
            path.unlink(missing_ok=True)
            removed += 1
        return removed

    def stats(self) -> dict[str, int]:
        """Return a per-namespace entry count, omitting empty namespaces.

        ``clear()`` leaves empty directories behind, so counting only
        non-empty namespaces keeps ``stats()`` a truthful picture of what is
        actually cached rather than of which directories happen to exist.
        """
        result: dict[str, int] = {}
        if not self.root.exists():
            return result
        for child in sorted(self.root.iterdir()):
            if not child.is_dir():
                continue
            count = sum(1 for _ in child.glob("*.json"))
            if count:
                result[child.name] = count
        return result


# Fixtures live in ``pipeline.synthetic``, not in a ``fixtures/`` data directory.
# That is a deliberate difference from the sibling MCP project: this pipeline's
# offline corpus is *generated* from a fixed seed rather than checked in, so there
# is no JSON to read and no fixture file to drift out of date.
