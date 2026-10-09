"""Normalisation: source-specific records -> :class:`~pipeline.models.Document`.

Live adapters and the offline corpus both emit plain ``dict`` records in one
shape, and this module is the single place that turns them into ``Document``.
That is deliberate: it means the *live* path and the *fixture* path execute
identical cleaning, id assignment and hashing code, so a passing offline run is
real evidence about production behaviour rather than evidence about a mock.

Boilerplate removal lives here too. Mining news pages carry subscribe prompts,
cookie banners, "read more" stubs and image captions; leaving them in inflates
every chunk and dilutes retrieval, and the eval set would measure how well the
ranker ignores boilerplate instead of how well it finds facts.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import datetime
from typing import Any

from pipeline.logging_setup import get_logger
from pipeline.models import Article, Document, PolicyItem, PricePoint, PriceSeries, SourceKind
from pipeline.render import render_day_document, render_summary_document
from pipeline.textutils import (
    content_hash,
    document_id,
    normalize_whitespace,
    parse_datetime,
    truncate,
)

__all__ = [
    "clean_text",
    "document_from_record",
    "record_from_article",
    "record_from_policy",
    "record_from_price_point",
    "record_from_price_series",
    "records_from_price_series",
]

_log = get_logger(__name__)

#: Lines that are pure navigation, promotion or legalese. Matched case-insensitively
#: against a whole normalised line, so a body sentence that merely mentions
#: "subscribe" is untouched.
_BOILERPLATE = re.compile(
    r"^\s*(?:"
    r"subscribe(?:\s+now)?|sign\s+up|newsletter|advertisement|advert|sponsored|"
    r"read\s+more|related\s+articles?|share\s+this\s+article|comments?|"
    r"cookie(?:s|\s+policy)?|privacy\s+policy|terms\s+of\s+(?:use|service)|"
    r"all\s+rights\s+reserved|copyright\s+©?\s*\d{4}.*|"
    r"follow\s+us(?:\s+on)?.*|download\s+the\s+app|"
    r"本页内容|版权所有|免责声明|免责条款|广告|订阅|相关阅读|上一篇|下一篇|返回顶部|"
    r"扫一扫|分享到|责任编辑[:：].*"
    r")\s*$",
    re.IGNORECASE,
)

#: Zero-width and direction-control characters. Present in scraped HTML, invisible
#: to a reader, and fatal to exact-content de-duplication.
_INVISIBLE = re.compile(r"[\u200b\u200c\u200d\u200e\u200f\u202a-\u202e\ufeff]")

#: An image caption or credit line left behind by HTML-to-text conversion.
_CAPTION = re.compile(
    r"^\s*(?:image|photo|credit|source|图|图片来源)\s*[:：].{0,120}$", re.IGNORECASE
)


def clean_text(text: str, *, max_chars: int = 40_000) -> str:
    """Normalise a document body: strip boilerplate, collapse whitespace, bound length.

    ``max_chars`` is a guard, not a feature: a malformed page that inlines an
    entire sitemap should not become a 5 MB chunk source.
    """
    if not text:
        return ""
    text = _INVISIBLE.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    kept: list[str] = []
    for raw_line in text.split("\n"):
        line = normalize_whitespace(raw_line)
        if not line:
            kept.append("")
            continue
        if _BOILERPLATE.match(line) or _CAPTION.match(line):
            continue
        kept.append(line)

    joined = "\n".join(kept)
    # Collapse runs of blank lines but keep paragraph structure: chunking works
    # better when paragraph boundaries survive.
    joined = re.sub(r"\n{3,}", "\n\n", joined).strip()
    return joined[:max_chars]


def _as_kind(value: object) -> SourceKind:
    if isinstance(value, SourceKind):
        return value
    try:
        return SourceKind(str(value))
    except ValueError as exc:
        raise ValueError(f"unknown source kind: {value!r}") from exc


def _as_datetime(value: object) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    return parse_datetime(str(value))


def document_from_record(record: Mapping[str, Any]) -> Document:
    """Build a :class:`Document` from a collector record.

    Raises ``ValueError`` on a missing required field. Collectors are responsible
    for producing well-formed records, and a loud failure here is much cheaper
    than a document with an empty body silently entering the index.
    """
    kind = _as_kind(record.get("kind"))
    title = normalize_whitespace(str(record.get("title") or ""))
    url = str(record.get("url") or "").strip()
    if not title:
        raise ValueError("record is missing a title")
    if not url:
        raise ValueError(f"record {title!r} is missing a url")

    raw_text = str(record.get("text") or "")
    # Fall back to the title so that a headline-only item is still retrievable
    # rather than being dropped.
    body = clean_text(raw_text) or title
    published = _as_datetime(record.get("published_at"))

    tags = record.get("tags") or []
    meta = record.get("meta") or {}
    return Document(
        id=document_id(kind.value, url, title, published),
        kind=kind,
        source=normalize_whitespace(str(record.get("source") or "unknown")),
        title=title,
        url=url,
        text=body,
        published_at=published,
        language=str(record.get("language") or "en"),
        jurisdiction=str(record.get("jurisdiction") or "").upper(),
        issuer=normalize_whitespace(str(record.get("issuer") or "")),
        doc_type=str(record.get("doc_type") or ""),
        commodity=str(record.get("commodity") or ""),
        tags=[str(tag) for tag in tags],
        meta=dict(meta) if isinstance(meta, Mapping) else {},
        content_hash=content_hash(body),
    )


# ---------------------------------------------------------------------------
# Typed source objects -> records
# ---------------------------------------------------------------------------


def record_from_article(article: Article) -> dict[str, Any]:
    """Record for the news source."""
    body_parts = [article.summary, article.content or ""]
    return {
        "kind": SourceKind.NEWS,
        "title": article.title,
        "url": article.url,
        "text": "\n".join(part for part in body_parts if part) or article.title,
        "published_at": article.published_at,
        "source": article.source,
        "language": article.language,
        "tags": ["news", *article.tags],
    }


def record_from_policy(item: PolicyItem) -> dict[str, Any]:
    """Record for the policy source."""
    body_parts = [item.summary, item.content or ""]
    return {
        "kind": SourceKind.POLICY,
        "title": item.title,
        "url": item.url,
        "text": "\n".join(part for part in body_parts if part) or item.title,
        "published_at": item.published_at,
        "source": item.issuer or "policy",
        "language": item.language,
        "jurisdiction": item.jurisdiction,
        "issuer": item.issuer,
        "doc_type": item.doc_type,
        "commodity": item.commodities[0] if item.commodities else "",
        "tags": ["policy", *item.commodities],
        "meta": {"effective_date": item.effective_date.isoformat() if item.effective_date else ""},
    }


def record_from_price_series(series: PriceSeries) -> dict[str, Any]:
    """Window-summary record for one commodity's price series."""
    last_point = series.points[-1] if series.points else None
    return {
        "kind": SourceKind.PRICE,
        "title": f"{series.commodity.value} price trend, {len(series.points)} sessions",
        "url": f"https://prices.example/{series.commodity.value}/trend-{len(series.points)}d",
        "text": render_summary_document(series),
        "published_at": last_point.date if last_point else None,
        "source": series.source or "price",
        "commodity": series.commodity.value,
        "tags": ["price", "trend", series.commodity.value],
        "meta": {
            "benchmark": series.benchmark,
            "unit": series.unit,
            "change_pct": series.change_pct,
            "direction": series.direction,
            "sessions": len(series.points),
        },
    }


def record_from_price_point(
    series: PriceSeries, point: PricePoint, *, index: int
) -> dict[str, Any]:
    """Daily-close record for one commodity."""
    return {
        "kind": SourceKind.PRICE,
        "title": f"{series.commodity.value} daily close {point.date.isoformat()}",
        "url": f"https://prices.example/{series.commodity.value}/{point.date.isoformat()}",
        "text": render_day_document(series, point, index=index),
        "published_at": point.date,
        "source": series.source or "price",
        "commodity": series.commodity.value,
        "tags": ["price", "daily", series.commodity.value],
        "meta": {
            "benchmark": series.benchmark,
            "unit": series.unit,
            "close": point.close,
            "date": point.date.isoformat(),
        },
    }


def records_from_price_series(
    series: PriceSeries, *, daily_days: int = 60, include_summary: bool = True
) -> list[dict[str, Any]]:
    """Every record derived from one commodity: the window summary plus daily closes."""
    records: list[dict[str, Any]] = []
    if include_summary:
        records.append(record_from_price_series(series))
    points = series.points[-daily_days:] if daily_days > 0 else []
    offset = len(series.points) - len(points)
    for index, point in enumerate(points):
        records.append(record_from_price_point(series, point, index=offset + index))
    return records


def summarize_document(document: Document, *, limit: int = 200) -> str:
    """A one-line description, used in logs and API payload previews."""
    return truncate(f"[{document.kind.value}/{document.source}] {document.title}", limit)
