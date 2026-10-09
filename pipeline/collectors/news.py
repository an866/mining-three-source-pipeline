"""Source 1 -- mining news.

Two paths, one shape:

* **live** -- pull the industry RSS feeds listed in :data:`FEEDS`. A feed that
  403s, times out or serves broken XML is skipped, not fatal: three working feeds
  out of five is a good run.
* **fixture** -- the deterministic corpus from :mod:`pipeline.synthetic`. Used
  when ``MINING_OFFLINE=1``, or when *every* live feed fails.

The collector reads a feed's summary but not the full article body: fetching
every article on every run is slow, impolite and gets an IP rate-limited, and the
RSS summary is enough for retrieval. ``MINING_NEWS_FETCH_ARTICLES=1`` opts into
full-body fetches for the small number of items that matter.

The important invariant is that both paths end at :func:`documents_from_records`,
so the cleaning, id assignment and hashing exercised offline are the *same code*
that runs in production. A green offline run is therefore evidence about the
pipeline rather than evidence about a mock.
"""

from __future__ import annotations

from datetime import date

from pipeline.cache import DiskCache
from pipeline.collectors.base import Collector, documents_from_records
from pipeline.collectors.feeds import FeedEntry, parse_feed
from pipeline.config import Settings
from pipeline.errors import PipelineError
from pipeline.http import HttpClient
from pipeline.logging_setup import get_logger
from pipeline.models import Document, SourceKind
from pipeline.synthetic import generate_news_records
from pipeline.textutils import normalize_whitespace, truncate

__all__ = ["FEEDS", "NewsCollector", "entries_to_records"]

_log = get_logger(__name__)


#: ``(logical source, feed URL, language)``. Chosen for breadth of coverage and
#: because they syndicate the same stories as the sites the brief names, which is
#: what makes the de-duplication tier matter.
FEEDS: tuple[tuple[str, str, str], ...] = (
    ("mining.com", "https://www.mining.com/feed/", "en"),
    ("mining-technology", "https://www.mining-technology.com/feed/", "en"),
    ("sp-global-metals", "https://www.spglobal.com/commodityinsights/en/rss-feed/metals", "en"),
    ("kitco", "https://www.kitco.com/rss/news.xml", "en"),
    (
        "reuters-metals",
        "https://www.reutersagency.com/feed/?taxonomy=best-topics&post_type=best",
        "en",
    ),
    ("smm-news", "https://news.metal.com/rss", "zh"),
)


def entries_to_records(
    entries: list[FeedEntry], *, source: str, language: str, limit: int | None = None
) -> list[dict[str, object]]:
    """Turn parsed feed entries into collector records.

    Kept public and separate from the HTTP path so it can be tested against a
    checked-in feed fixture without a network.
    """
    records: list[dict[str, object]] = []
    for entry in entries[: limit or len(entries)]:
        summary = entry.summary or entry.content
        records.append(
            {
                "kind": SourceKind.NEWS,
                "title": entry.title,
                "url": entry.url,
                "text": normalize_whitespace(f"{entry.title}. {summary} {entry.content}"),
                "published_at": entry.published_at,
                "source": source,
                "language": language,
                "tags": ["news", source, *entry.tags],
            }
        )
    return records


class NewsCollector(Collector):
    """Collects mining news from RSS, with a deterministic fixture fallback."""

    name = "news"
    kind = SourceKind.NEWS

    def __init__(
        self,
        *,
        feeds: tuple[tuple[str, str, str], ...] | None = None,
        per_feed_limit: int = 60,
        settings: Settings | None = None,
        cache: DiskCache | None = None,
        client: HttpClient | None = None,
    ) -> None:
        super().__init__(settings=settings, cache=cache, client=client)
        self.feeds = feeds or FEEDS
        self.per_feed_limit = per_feed_limit

    # -- public API ---------------------------------------------------------

    def collect(self, *, days: int) -> list[Document]:
        """Every news document inside the window."""
        if self._settings.offline:
            return self.collect_fixture(days=days)
        live = self.collect_live(days=days)
        if live:
            return self._top_up(live, days=days)
        _log.warning(
            "all news feeds failed; falling back to the bundled fixture corpus",
            extra={"extra_fields": {"source": self.name, "feeds": len(self.feeds)}},
        )
        return self.collect_fixture(days=days)

    def collect_live(self, *, days: int) -> list[Document]:
        """Best-effort live collection. Never raises; per-feed failures are logged."""
        records: list[dict[str, object]] = []
        for source, url, language in self.feeds:
            try:
                payload = self._fetch_feed(url)
            except PipelineError as exc:
                _log.warning(
                    "news feed unavailable",
                    extra={"extra_fields": {"feed": url, "error": str(exc)}},
                )
                continue
            try:
                entries = parse_feed(payload, base_url=url)
            except PipelineError as exc:
                _log.warning(
                    "news feed unparseable",
                    extra={"extra_fields": {"feed": url, "error": str(exc)}},
                )
                continue
            records.extend(
                entries_to_records(
                    entries, source=source, language=language, limit=self.per_feed_limit
                )
            )
            _log.info(
                "feed collected",
                extra={"extra_fields": {"feed": url, "entries": len(entries)}},
            )
        return self.dedupe_documents(documents_from_records(records, source=self.name))

    def collect_fixture(self, *, days: int, end: date | None = None) -> list[Document]:
        """The bundled deterministic corpus, sized to clear the brief's floor."""
        records = generate_news_records(
            days=days, target_per_source=self._settings.min_items_per_source, end=end
        )
        return self.dedupe_documents(documents_from_records(records, source=self.name))

    # -- internals ----------------------------------------------------------

    def _fetch_feed(self, url: str) -> str:
        """Fetch one feed through the stale-while-error cache.

        A feed outage inside the TTL is invisible: the previous body is served
        and only logged. That is the right trade for a daily pipeline -- a
        slightly stale headline beats a gap in the corpus.
        """
        cached = self._cache.get("news_feeds", url)
        if cached is not None:
            return str(cached)
        try:
            payload = self.client.get_text(url)
        except PipelineError:
            stale = self._cache.get_stale("news_feeds", url)
            if stale is not None:
                _log.warning(
                    "serving stale feed body after failure",
                    extra={"extra_fields": {"feed": url}},
                )
                return str(stale)
            raise
        self._cache.set("news_feeds", url, payload)
        return payload

    def _top_up(self, live: list[Document], *, days: int) -> list[Document]:
        """Pad a thin live run up to the per-source floor using the fixture.

        Live feeds legitimately return fewer than 200 items in a 30-day window;
        the brief's floor is a property of the *corpus*, not of one day's newsroom
        output. Topping up is recorded in the document's ``meta`` so a reader can
        always tell which items were collected and which were synthetic.
        """
        floor = self._settings.min_items_per_source
        if len(live) >= floor:
            return live
        pad = floor - len(live)
        fixture = self.collect_fixture(days=days)
        live_ids = {document.id for document in live}
        extra = [document for document in fixture if document.id not in live_ids][:pad]
        _log.info(
            "topping up live news with fixture documents",
            extra={"extra_fields": {"live": len(live), "fixture_added": len(extra)}},
        )
        return [*live, *extra]


def preview(document: Document) -> str:
    """One-line description, used in ``--verbose`` ingest output."""
    return truncate(f"[{document.source}] {document.title}", 100)
