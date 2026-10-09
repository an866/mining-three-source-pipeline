"""Collector contract.

Every source, live or fixture, is a :class:`Collector` that returns a list of
:class:`~pipeline.models.Document`. Normalising to one shape here is what keeps
cleaning, de-duplication, chunking, embedding and indexing source-agnostic: those
stages never branch on "am I looking at news or at a price table".

Two rules the subclasses must respect:

* **``days`` is a hard window.** A collector that returns older material makes the
  "近 7 天" filters in the API silently wrong.
* **``safe_collect`` never raises.** A source that is down degrades to zero
  documents plus a message. The pipeline run is allowed to be partial; it is not
  allowed to abort because one of three sources returned a 403.
"""

from __future__ import annotations

import abc
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime, timedelta
from typing import Any

from pipeline.cache import DiskCache
from pipeline.config import Settings, get_settings
from pipeline.errors import PipelineError
from pipeline.http import HttpClient
from pipeline.logging_setup import get_logger
from pipeline.models import Document, SourceKind
from pipeline.normalize import document_from_record

__all__ = ["Collector", "SourceReport", "documents_from_records", "in_window"]

_log = get_logger(__name__)


class SourceReport:
    """Per-source outcome of one run, for the ingest report.

    ``dropped_identity`` exists so that the run's duplicate accounting adds up.
    The source-level identity guard and the cascade's tier 1 use the same key, so
    whichever runs first takes the removal -- and reporting only the cascade's count
    would understate how much duplication the run actually removed.
    """

    __slots__ = ("documents", "dropped_identity", "dropped_out_of_window", "error", "name")

    def __init__(
        self,
        name: str,
        documents: list[Document],
        error: str = "",
        *,
        dropped_identity: int = 0,
        dropped_out_of_window: int = 0,
    ) -> None:
        self.name = name
        self.documents = documents
        self.error = error
        self.dropped_identity = dropped_identity
        self.dropped_out_of_window = dropped_out_of_window

    @property
    def ok(self) -> bool:
        return not self.error

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"SourceReport(name={self.name!r}, documents={len(self.documents)}, "
            f"dropped_identity={self.dropped_identity}, error={self.error!r})"
        )


def in_window(published: datetime | None, *, days: int, now: datetime | None = None) -> bool:
    """Whether ``published`` falls inside the last ``days`` days.

    A document with no date is **kept**: dropping undated material silently
    removes the newest items on feeds that omit ``pubDate``, and a false positive
    is cheaper than a missing document.
    """
    if published is None:
        return True
    reference = now or datetime.now(UTC)
    if published.tzinfo is None:
        published = published.replace(tzinfo=UTC)
    return published >= reference - timedelta(days=days)


def documents_from_records(
    records: Iterable[Mapping[str, Any]], *, source: str = ""
) -> list[Document]:
    """Convert collector records to :class:`Document`, skipping malformed ones.

    One bad record must not cost a whole source. A record with no title or no URL
    is dropped with a warning; everything else is preserved. Note that this is
    *not* a place to be lenient about malformed *content* -- that would let the
    index fill with empty bodies, which is worse than a short corpus.
    """
    documents: list[Document] = []
    dropped = 0
    for record in records:
        try:
            documents.append(document_from_record(record))
        except ValueError as exc:
            dropped += 1
            _log.warning(
                "dropping malformed record",
                extra={"extra_fields": {"source": source or "unknown", "error": str(exc)}},
            )
    if dropped:
        _log.warning(
            "records dropped during normalisation",
            extra={"extra_fields": {"source": source or "unknown", "dropped": dropped}},
        )
    return documents


class Collector(abc.ABC):
    """Base class for the three source adapters."""

    #: Adapter name, also used as ``Document.source``.
    name: str = "abstract"
    #: Which of the three source families this adapter belongs to.
    kind: SourceKind = SourceKind.NEWS
    #: Minimum collection horizon, in days, regardless of the requested window.
    #:
    #: Sources do not all need the same history. News and policy are consumed
    #: "as of now" -- a 30-day window is the whole point. Prices are consumed as a
    #: *series*: a trend question needs history behind it, and a 30-day window of
    #: weekdays yields barely twenty observations, which is too few to characterise
    #: a move. So the price collector overrides this to keep roughly three months
    #: of sessions. This is why the price source legitimately contributes more
    #: documents than the other two, and why its 30-day figures are computed from a
    #: longer series rather than truncated.
    history_days: int = 0

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        cache: DiskCache | None = None,
        client: HttpClient | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._cache = cache if cache is not None else DiskCache(settings=self._settings)
        self._client = client
        #: Duplicates the source-level identity guard removed. Reset per collect.
        self.dropped_identity = 0

    # -- template methods ---------------------------------------------------

    @abc.abstractmethod
    def collect(self, *, days: int) -> list[Document]:
        """Return every document this source published in the last ``days`` days."""

    def safe_collect(self, *, days: int) -> SourceReport:
        """Run :meth:`collect`, turning any failure into an empty report."""
        window = max(days, self.history_days)
        try:
            documents = self.collect(days=window)
        except PipelineError as exc:
            _log.warning(
                "collector failed",
                extra={"extra_fields": {"source": self.name, "error": str(exc)}},
            )
            return SourceReport(self.name, [], f"{type(exc).__name__}: {exc}")
        except Exception as exc:
            _log.error(
                "collector crashed",
                extra={"extra_fields": {"source": self.name, "error": str(exc)}},
            )
            return SourceReport(self.name, [], f"{type(exc).__name__}: {exc}")
        kept = [doc for doc in documents if in_window(doc.published_at, days=window)]
        dropped_window = len(documents) - len(kept)
        _log.info(
            "collected",
            extra={
                "extra_fields": {
                    "source": self.name,
                    "documents": len(kept),
                    "window_days": window,
                    "dropped_out_of_window": dropped_window,
                    "dropped_identity": self.dropped_identity,
                }
            },
        )
        return SourceReport(
            self.name,
            kept,
            dropped_identity=self.dropped_identity,
            dropped_out_of_window=dropped_window,
        )

    # -- helpers ------------------------------------------------------------

    @property
    def client(self) -> HttpClient:
        """A lazily constructed HTTP client, bound to this collector's source name."""
        if self._client is None:
            self._client = HttpClient(source=self.name, settings=self._settings)
        return self._client

    def close(self) -> None:
        """Release the HTTP connection pool, if one was opened."""
        if self._client is not None:
            self._client.close()
            self._client = None

    def dedupe_documents(self, documents: Iterable[Document]) -> list[Document]:
        """Collapse duplicates by primary key, first occurrence winning.

        This is the *within-source* identity guard, and it is deliberately separate
        from :class:`~pipeline.dedup.Deduper`. Its job is to keep the per-source
        count meaningful before the floor check in ``_top_up`` -- a feed that
        repeats an item must not make a thin run look complete. The cascade's tier 1
        has the same key and therefore rarely sees a collision: on the offline
        corpus the guard takes all six injected ``recollected`` copies per source and
        the cascade reports ``exact_id: 0``. The drop count is logged here so the
        total duplication removed by the run adds up.
        """
        incoming = list(documents)
        seen: set[str] = set()
        unique: list[Document] = []
        for document in incoming:
            if document.id in seen:
                continue
            seen.add(document.id)
            unique.append(document)
        removed = len(incoming) - len(unique)
        if removed:
            self.dropped_identity += removed
            _log.info(
                "source-level identity guard removed duplicates",
                extra={"extra_fields": {"source": self.name, "removed": removed}},
            )
        return unique
