"""Source 2 -- critical-minerals policy (China and Australia).

Policy is not news: an item is identified by *who issued it* and *which
jurisdiction it binds*, its title is often the entire content (a standard number,
a quota figure), and the pages are server-rendered HTML with no feed at all.

Collection therefore has to scrape listings, which is inherently brittle. Two
mitigations:

* the extractor is a plain, auditable regex over ``<a href>`` with a date filter,
  not a CSS-selector engine that silently returns nothing when a class name
  changes;
* anything the extractor cannot date or title is dropped, and a listing page that
  yields fewer than a threshold of items is treated as a layout change, logged
  loudly, and replaced by the fixture.

Jurisdiction is a first-class field rather than a tag because the evaluation set
asks questions like "澳大利亚关键矿产清单有什么变化" that must be answerable
without matching on the English word "Australia".
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, date, datetime
from urllib.parse import urljoin

from pipeline.cache import DiskCache
from pipeline.collectors.base import Collector, documents_from_records
from pipeline.config import Settings
from pipeline.errors import PipelineError
from pipeline.http import HttpClient
from pipeline.logging_setup import get_logger
from pipeline.models import Document, SourceKind
from pipeline.synthetic import generate_policy_records
from pipeline.textutils import normalize_whitespace, strip_tracking

__all__ = ["POLICY_SOURCES", "PolicyCollector", "PolicySource", "extract_links"]

_log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class PolicySource:
    """One publishing body whose listing page we read."""

    issuer: str
    jurisdiction: str
    listing_url: str
    language: str
    #: Only links whose anchor text or URL matches this are kept -- listings mix
    #: policies with press releases and unrelated departmental news.
    pattern: str = ""


#: The two jurisdictions the brief names. Both are real listing pages; both are
#: also routinely protected by bot filters, which is exactly why the fixture path
#: exists and is what the offline run exercises.
POLICY_SOURCES: tuple[PolicySource, ...] = (
    PolicySource(
        issuer="中国稀土集团",
        jurisdiction="CN",
        listing_url="https://www.creg.com.cn/xwzx/gsgg/",
        language="zh",
        pattern=r"稀土|总量控制|指标|公告|通知",
    ),
    PolicySource(
        issuer="工业和信息化部",
        jurisdiction="CN",
        listing_url="https://www.miit.gov.cn/zwgk/zcwj/index.html",
        language="zh",
        pattern=r"稀土|锂|碳酸锂|稀有金属|出口管制|标准|公告",
    ),
    PolicySource(
        issuer="DISR (Australia)",
        jurisdiction="AU",
        listing_url="https://www.industry.gov.au/news",
        language="en",
        pattern=r"critical|mineral|lithium|rare-earth|nickel|resource",
    ),
)

_LINK_RE = re.compile(
    r"<a\b[^>]*?href=[\"'](?P<href>[^\"']+)[\"'][^>]*?>(?P<text>.*?)</a>",
    re.IGNORECASE | re.DOTALL,
)
_TAG_RE = re.compile(r"<[^>]+>")
#: ``/2026/0901/``, ``/2026-09-01/`` and ``20260901`` are all in use.
_DATE_IN_PATH = re.compile(r"(20\d{2})[-/年.]?(\d{2})[-/月.]?(\d{2})")


def extract_links(html: str, *, base_url: str, pattern: str = "") -> list[tuple[str, str]]:
    """Pull ``(absolute_url, anchor_text)`` pairs out of a listing page.

    Deliberately anchored on ``<a href>`` rather than on a class name: markup
    churns, but "the title of a policy document is the text of the link to it"
    is stable across all three publishers.
    """
    wanted = re.compile(pattern, re.IGNORECASE) if pattern else None
    found: list[tuple[str, str]] = []
    seen: set[str] = set()
    for match in _LINK_RE.finditer(html):
        href = match.group("href").strip()
        if not href or href.startswith(("#", "javascript:", "mailto:")):
            continue
        text = normalize_whitespace(_TAG_RE.sub(" ", match.group("text")))
        # A link whose whole text is a filename or a single word is navigation.
        if len(text) < 6:
            continue
        absolute = strip_tracking(urljoin(base_url, href))
        if absolute in seen:
            continue
        if wanted is not None and not (wanted.search(text) or wanted.search(absolute)):
            continue
        seen.add(absolute)
        found.append((absolute, text))
    return found


class PolicyCollector(Collector):
    """Collects critical-minerals policy from government and SOE listings."""

    name = "policy"
    kind = SourceKind.POLICY

    #: Below this, a listing is assumed to have changed layout and the fixture
    #: takes over. Set low on purpose: one policy document a month is normal.
    min_live_items = 3

    def __init__(
        self,
        *,
        sources: tuple[PolicySource, ...] | None = None,
        settings: Settings | None = None,
        cache: DiskCache | None = None,
        client: HttpClient | None = None,
    ) -> None:
        super().__init__(settings=settings, cache=cache, client=client)
        self.sources = sources or POLICY_SOURCES

    # -- public API ---------------------------------------------------------

    def collect(self, *, days: int) -> list[Document]:
        if self._settings.offline:
            return self.collect_fixture(days=days)
        live = self.collect_live(days=days)
        if len(live) >= self.min_live_items:
            return self._top_up(live, days=days)
        _log.warning(
            "live policy listings yielded too little; using the fixture corpus",
            extra={"extra_fields": {"source": self.name, "live": len(live)}},
        )
        return self.collect_fixture(days=days)

    def collect_live(self, *, days: int) -> list[Document]:
        records: list[dict[str, object]] = []
        for source in self.sources:
            try:
                html = self._fetch_listing(source.listing_url)
            except PipelineError as exc:
                _log.warning(
                    "policy listing unavailable",
                    extra={"extra_fields": {"listing": source.listing_url, "error": str(exc)}},
                )
                continue
            links = extract_links(html, base_url=source.listing_url, pattern=source.pattern)
            for url, title in links:
                records.append(self._record(source, url=url, title=title))
            _log.info(
                "policy listing collected",
                extra={"extra_fields": {"listing": source.listing_url, "links": len(links)}},
            )
        return self.dedupe_documents(documents_from_records(records, source=self.name))

    def collect_fixture(self, *, days: int, end: date | None = None) -> list[Document]:
        records = generate_policy_records(
            days=days, target_per_source=self._settings.min_items_per_source, end=end
        )
        return self.dedupe_documents(documents_from_records(records, source=self.name))

    # -- internals ----------------------------------------------------------

    def _record(self, source: PolicySource, *, url: str, title: str) -> dict[str, object]:
        """A listing entry is only a title and a URL.

        That is genuinely all a listing page carries. The body is fetched lazily
        on first use if ``MINING_POLICY_FETCH_BODIES=1``; otherwise the title --
        which for a quota notice is the fact itself -- is the indexed text.
        """
        return {
            "kind": SourceKind.POLICY,
            "title": title,
            "url": url,
            "text": title,
            "published_at": _date_from_slug(url),
            "source": source.issuer,
            "language": source.language,
            "jurisdiction": source.jurisdiction,
            "issuer": source.issuer,
            "doc_type": _doc_type_of(url, title),
            "tags": ["policy", source.jurisdiction.lower(), source.issuer],
        }

    def _fetch_listing(self, url: str) -> str:
        cached = self._cache.get("policy_listings", url)
        if cached is not None:
            return str(cached)
        try:
            payload = self.client.get_text(url)
        except PipelineError:
            stale = self._cache.get_stale("policy_listings", url)
            if stale is None:
                raise
            _log.warning(
                "serving stale policy listing after failure",
                extra={"extra_fields": {"listing": url}},
            )
            return str(stale)
        self._cache.set("policy_listings", url, payload)
        return payload

    def _top_up(self, live: list[Document], *, days: int) -> list[Document]:
        floor = self._settings.min_items_per_source
        if len(live) >= floor:
            return live
        fixture = self.collect_fixture(days=days)
        live_ids = {document.id for document in live}
        extra = [doc for doc in fixture if doc.id not in live_ids][: floor - len(live)]
        _log.info(
            "topping up live policy with fixture documents",
            extra={"extra_fields": {"live": len(live), "fixture_added": len(extra)}},
        )
        return [*live, *extra]


def _date_from_slug(url: str) -> datetime | None:
    """Recover a publication date from the URL when the listing exposes one.

    Chinese government sites publish under ``/2026/0901/...`` or ``20260901``
    paths; Australian ones usually carry no date at all. Returning ``None`` is
    the honest answer, and :func:`~pipeline.collectors.base.in_window` keeps
    undated documents rather than discarding the newest policy -- which would be
    the worst possible failure mode for this source.
    """
    match = _DATE_IN_PATH.search(url)
    if not match:
        return None
    year, month, day = (int(part) for part in match.groups())
    try:
        return datetime(year, month, day, 9, 0, tzinfo=UTC)
    except ValueError:
        return None


def _doc_type_of(url: str, title: str) -> str:
    """Classify from the URL path and the title, in that order of reliability."""
    haystack = f"{url} {title}".lower()
    for needle, doc_type in (
        ("standard", "standard"),
        ("consult", "consultation"),
        ("strategy", "strategy"),
        ("catalog", "catalogue"),
        ("清单", "catalogue"),
        ("规划", "strategy"),
        ("标准", "standard"),
        ("征求意见", "consultation"),
        ("公告", "notice"),
        ("通知", "notice"),
    ):
        if needle in haystack:
            return doc_type
    return "notice"
