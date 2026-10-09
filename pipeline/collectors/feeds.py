"""A dependency-free RSS 2.0 / Atom 1.0 reader.

``feedparser`` is the obvious choice and is deliberately not used: it is a large
dependency for the small subset this pipeline needs (title, link, date, summary),
and it happily accepts malformed input, which hides feed breakage instead of
surfacing it. Twenty lines of :mod:`xml.etree` do the job and fail loudly.

The parsing logic is kept here, apart from the collector, because it is the part
that genuinely needs unit tests: namespace handling, three date formats, and
relative-link resolution all have edge cases that a fixture can pin down.
"""

from __future__ import annotations

import html
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime
from urllib.parse import urljoin

from pipeline.errors import ParseError
from pipeline.textutils import normalize_whitespace, parse_datetime

__all__ = ["FeedEntry", "parse_feed", "strip_html"]

#: Feeds are namespaced inconsistently in the wild; match on the local name and
#: ignore the prefix entirely.
_ATOM = "{http://www.w3.org/2005/Atom}"
_RSS_CONTENT = "{http://purl.org/rss/1.0/modules/content/}"
_DC = "{http://purl.org/dc/elements/1.1/}"

_TAG_RE = re.compile(r"<[^>]+>")
_SCRIPT_RE = re.compile(r"<(script|style)\b.*?</\1>", re.IGNORECASE | re.DOTALL)
_ENTITY_RE = re.compile(r"&[a-zA-Z]+;|&#\d+;")


@dataclass(slots=True)
class FeedEntry:
    """One item from a feed, before the collector adds source-level metadata."""

    title: str
    url: str
    published_at: datetime | None = None
    summary: str = ""
    content: str = ""
    tags: list[str] = field(default_factory=list)


def strip_html(value: str) -> str:
    """Reduce an HTML fragment to plain text.

    Feeds routinely embed markup inside ``<description>``; leaving tags in place
    would put ``<p>`` and ``&amp;`` into the embedded text and into every
    citation. Entities are decoded twice in a row so that ``&amp;lt;`` (a literal
    ``&lt;`` in the source) survives exactly once.
    """
    if not value:
        return ""
    without_scripts = _SCRIPT_RE.sub(" ", value)
    text = _TAG_RE.sub(" ", without_scripts)
    text = _ENTITY_RE.sub(lambda match: html.unescape(match.group(0)), text)
    text = html.unescape(text)
    return normalize_whitespace(text)


def _first(element: ET.Element, *names: str) -> ET.Element | None:
    """The first child matching any of ``names``, searching one level deep."""
    for name in names:
        found = element.find(name)
        if found is not None:
            return found
    return None


def _text(element: ET.Element | None) -> str:
    if element is None:
        return ""
    return normalize_whitespace("".join(element.itertext()))


def _link_of(entry: ET.Element) -> str:
    """Resolve an item's URL across RSS and Atom conventions.

    Atom expresses the canonical link as ``<link rel="alternate" href="...">``
    while RSS uses ``<link>text</link>``; some feeds emit both, and some emit
    several ``<link>`` elements where only one is the human-facing page.
    """
    alternate = ""
    fallback = ""
    for link in entry.findall(f"{_ATOM}link") + entry.findall("link"):
        href = link.get("href") or _text(link)
        rel = (link.get("rel") or "alternate").lower()
        if href and rel == "alternate" and not alternate:
            alternate = href
        elif href and not fallback:
            fallback = href
    if not alternate and not fallback:
        # Some RSS feeds use a bare <guid isPermaLink="true">
        guid = entry.find("guid")
        if guid is not None and guid.get("isPermaLink", "false").lower() == "true":
            fallback = _text(guid)
    return alternate or fallback


def parse_feed(xml_text: str, *, base_url: str = "") -> list[FeedEntry]:
    """Parse an RSS or Atom document into :class:`FeedEntry` objects.

    Raises :class:`ParseError` when the payload is not well-formed XML. An empty
    channel is *not* an error -- a feed that carries nothing today is normal.
    """
    payload = xml_text.lstrip("\ufeff \t\r\n")
    try:
        root = ET.fromstring(payload)
    except ET.ParseError as exc:
        raise ParseError("feed", str(exc)) from exc

    # RSS items or Atom entries; the annotation is inferred rather than written,
    # because ``root.iter`` yields ``Element``, not ``FeedEntry``.
    entries = list(root.iter("item")) or list(root.iter(f"{_ATOM}entry"))
    parsed: list[FeedEntry] = []
    for entry in entries:
        title = strip_html(_text(_first(entry, "title", f"{_ATOM}title")))
        raw_link = _link_of(entry)
        url = urljoin(base_url, raw_link) if base_url else raw_link
        if not title or not url:
            # A headline-less or link-less item is unusable downstream and
            # dropping it here is cheaper than a validation error later.
            continue

        summary = strip_html(_text(_first(entry, "description", "summary", f"{_ATOM}summary")))
        content = strip_html(
            _text(_first(entry, f"{_RSS_CONTENT}encoded", "content", f"{_ATOM}content"))
        )
        published = parse_datetime(
            _text(
                _first(
                    entry,
                    "pubDate",
                    "published",
                    f"{_ATOM}published",
                    f"{_ATOM}updated",
                    "updated",
                    f"{_DC}date",
                )
            )
        )
        tags = [
            normalize_whitespace(_text(category))
            for category in entry.findall("category") + entry.findall(f"{_ATOM}category")
            if _text(category)
        ]
        parsed.append(
            FeedEntry(
                title=title,
                url=url,
                published_at=published,
                summary=summary,
                content=content,
                tags=tags,
            )
        )
    return parsed
