"""Domain models for the three-source pipeline.

One type per concept, shared by the collectors, the vector store, the REST
service and the evaluation harness. Three rules hold throughout:

* ``extra="forbid"`` -- a collector that invents a field fails at the boundary
  instead of silently writing junk into the index.
* ids are deterministic content hashes, never ``uuid4`` -- re-running the
  pipeline over the same data must not create new rows (see DATA_NOTES.md).
* derived values (price statistics, chunk offsets) are computed once, by the
  producer, so the API and the eval harness cannot disagree about them.
"""

from __future__ import annotations

from datetime import date as Date  # noqa: N812 -- `date` is used as a field name
from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

__all__ = [
    "COMMODITY_META",
    "Article",
    "Chunk",
    "Citation",
    "Commodity",
    "Document",
    "IngestStats",
    "PolicyItem",
    "PricePoint",
    "PriceQuote",
    "PriceSeries",
    "QueryAnswer",
    "QueryFilters",
    "QueryRequest",
    "RetrievedChunk",
    "SourceKind",
]


class _Model(BaseModel):
    """Project-wide base model: strict-ish, JSON-friendly."""

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        str_strip_whitespace=True,
        validate_assignment=False,
    )


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class SourceKind(StrEnum):
    """The three source families required by the brief."""

    NEWS = "news"
    POLICY = "policy"
    PRICE = "price"


class Commodity(StrEnum):
    """Commodities the price source covers.

    Deliberately the exact five named in the brief: LME copper/zinc/nickel,
    SHFE lithium carbonate, and Mysteel 62% Fe iron ore.
    """

    COPPER = "copper"
    ZINC = "zinc"
    NICKEL = "nickel"
    LITHIUM = "lithium"
    IRON_ORE = "iron_ore"


COMMODITY_META: dict[Commodity, dict[str, str]] = {
    Commodity.COPPER: {
        "unit": "USD/t",
        "currency": "USD",
        "benchmark": "LME 3M Copper",
        "exchange": "LME",
        "zh": "铜",
    },
    Commodity.ZINC: {
        "unit": "USD/t",
        "currency": "USD",
        "benchmark": "LME 3M Zinc",
        "exchange": "LME",
        "zh": "锌",
    },
    Commodity.NICKEL: {
        "unit": "USD/t",
        "currency": "USD",
        "benchmark": "LME 3M Nickel",
        "exchange": "LME",
        "zh": "镍",
    },
    Commodity.LITHIUM: {
        "unit": "CNY/t",
        "currency": "CNY",
        "benchmark": "SHFE Lithium Carbonate",
        "exchange": "SHFE",
        "zh": "碳酸锂",
    },
    Commodity.IRON_ORE: {
        "unit": "CNY/t",
        "currency": "CNY",
        "benchmark": "Mysteel 62% Fe CFR Qingdao",
        "exchange": "Mysteel",
        "zh": "铁矿石",
    },
}


# ---------------------------------------------------------------------------
# Source 1 -- mining news
# ---------------------------------------------------------------------------


class Article(_Model):
    """A news item, normalised across every feed that carries it."""

    id: str = Field(description="Stable content hash; the de-duplication key.")
    title: str
    url: str
    source: str = Field(description="Logical feed name, e.g. 'mining.com'.")
    published_at: datetime | None = None
    summary: str = ""
    content: str | None = Field(default=None, description="Full body text, when fetched.")
    language: str = "en"
    tags: list[str] = Field(default_factory=list)

    @field_validator("url")
    @classmethod
    def _absolute_url(cls, value: str) -> str:
        if not value.startswith(("http://", "https://")):
            raise ValueError(f"article url must be absolute, got {value!r}")
        return value


# ---------------------------------------------------------------------------
# Source 2 -- critical-minerals policy
# ---------------------------------------------------------------------------


class PolicyItem(_Model):
    """A policy document from a government body or a state-owned producer.

    Kept separate from :class:`Article` because the fields that matter differ:
    a policy item is identified by *who issued it* and *which jurisdiction it
    binds*, not by which feed syndicated it.
    """

    id: str
    title: str
    url: str
    issuer: str = Field(description="e.g. 'China Rare Earth Group', 'DISR (Australia)'.")
    jurisdiction: str = Field(default="", description="ISO-3166 alpha-2, e.g. 'CN', 'AU'.")
    doc_type: str = Field(
        default="",
        description="notice | strategy | standard | catalogue | speech | consultation",
    )
    published_at: datetime | None = None
    summary: str = ""
    content: str | None = None
    language: str = "en"
    commodities: list[str] = Field(
        default_factory=list, description="Commodities the document concerns."
    )
    effective_date: Date | None = Field(
        default=None, description="When the instrument takes effect, if stated."
    )

    @field_validator("url")
    @classmethod
    def _absolute_url(cls, value: str) -> str:
        if not value.startswith(("http://", "https://")):
            raise ValueError(f"policy url must be absolute, got {value!r}")
        return value


# ---------------------------------------------------------------------------
# Source 3 -- prices
# ---------------------------------------------------------------------------


class PricePoint(_Model):
    """One OHLC observation for a commodity on a given trading day."""

    date: Date
    close: float
    open: float | None = None
    high: float | None = None
    low: float | None = None
    volume: float | None = None


class PriceQuote(_Model):
    """A single-date quotation."""

    commodity: Commodity
    date: Date
    price: float
    currency: str = "USD"
    unit: str = "t"
    benchmark: str = ""
    source: str = ""
    is_estimated: bool = Field(
        default=False,
        description="True when the requested date had no session and the nearest one was used.",
    )


class PriceSeries(_Model):
    """An ordered series plus derived statistics.

    The statistics are computed once by the price collector rather than by each
    consumer, so a chart and an answer can never disagree about the change.
    """

    commodity: Commodity
    currency: str
    unit: str
    benchmark: str = ""
    source: str = ""
    points: list[PricePoint] = Field(default_factory=list)

    start_price: float | None = None
    end_price: float | None = None
    change_abs: float | None = None
    change_pct: float | None = None
    min_price: float | None = None
    max_price: float | None = None
    direction: Literal["up", "down", "flat"] = "flat"


# ---------------------------------------------------------------------------
# The unified ingest unit
# ---------------------------------------------------------------------------


class Document(_Model):
    """One ingestable unit, normalised across all three sources.

    Collectors emit ``Document``, never ``Article``/``PolicyItem``/``PriceSeries``
    directly, so cleaning, de-duplication, chunking and embedding have exactly
    one input shape to handle. ``text`` is what gets embedded -- for prices that
    means a rendered sentence, which is what makes "近一个月锂价怎么走" retrievable
    from a numeric table.
    """

    id: str = Field(description="Primary key: hash of the normalised text + identity.")
    kind: SourceKind
    source: str = Field(description="Adapter that produced it, e.g. 'mining.com'.")
    title: str
    url: str
    text: str = Field(description="Cleaned body text used for embedding and answering.")
    published_at: datetime | None = None
    language: str = "en"

    # Optional, source-specific. Empty string means "not applicable".
    jurisdiction: str = ""
    issuer: str = ""
    doc_type: str = ""
    commodity: str = ""

    tags: list[str] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)

    content_hash: str = Field(default="", description="Fingerprint of ``text`` after cleaning.")


class Chunk(_Model):
    """A retrievable span of a :class:`Document`.

    Denormalises the filter fields (kind / jurisdiction / commodity / date) so
    that filtered search never needs a join back to the document table.
    """

    id: str = Field(description="``<document_id>:<ordinal>``.")
    document_id: str
    ordinal: int = Field(ge=0)
    text: str
    char_start: int = 0
    char_end: int = 0

    kind: SourceKind
    source: str
    url: str
    title: str
    published_at: datetime | None = None
    jurisdiction: str = ""
    commodity: str = ""


# ---------------------------------------------------------------------------
# Query I/O
# ---------------------------------------------------------------------------


class QueryFilters(_Model):
    """Structured constraints extracted from (or supplied with) a question."""

    kinds: list[SourceKind] = Field(default_factory=list)
    days: int | None = Field(
        default=None, ge=1, le=3650, description="Recency window, e.g. 7 for '近 7 天'."
    )
    jurisdiction: str = ""
    source: str = ""
    commodity: str = ""

    def is_empty(self) -> bool:
        """True when no constraint is set, i.e. search everything."""
        return not any((self.kinds, self.days, self.jurisdiction, self.source, self.commodity))


class QueryRequest(_Model):
    """Body of ``POST /query``."""

    question: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=50)
    filters: QueryFilters = Field(default_factory=QueryFilters)
    synthesise: bool = Field(
        default=True, description="False returns retrieved evidence without an answer."
    )


class RetrievedChunk(_Model):
    """One search hit."""

    chunk: Chunk
    score: float
    rank: int = Field(ge=1, description="1-based rank in the result list.")


class Citation(_Model):
    """A numbered source reference resolving to a retrieved chunk."""

    index: int = Field(ge=1)
    kind: SourceKind
    title: str
    url: str
    published_at: datetime | None = None
    score: float = 0.0


class QueryAnswer(_Model):
    """Response of ``POST /query``."""

    question: str
    answer: str
    mode: Literal["llm", "extractive", "refusal"] = Field(
        description="'extractive' when no LLM key is configured; the answer is then "
        "assembled from retrieved spans and is fully traceable. 'refusal' means the "
        "question fell outside what the corpus covers and no evidence was returned."
    )
    citations: list[Citation] = Field(default_factory=list)
    retrieved: list[RetrievedChunk] = Field(default_factory=list)
    filters: QueryFilters = Field(default_factory=QueryFilters)
    latency_ms: int = 0
    offline: bool = False


# ---------------------------------------------------------------------------
# Operational reports
# ---------------------------------------------------------------------------


class IngestStats(_Model):
    """Outcome of one pipeline run, per source."""

    documents: int = 0
    chunks: int = 0
    per_source: dict[str, int] = Field(default_factory=dict)
    per_kind: dict[str, int] = Field(default_factory=dict)
    #: Total duplicate documents the run removed, across every layer. This is the
    #: headline number and it is the *whole* number: quoting only the cascade's share
    #: would understate a run whose source-level identity guard took the rest.
    duplicates_removed: int = 0
    #: Where those removals happened, e.g.
    #: ``{"collector_identity": 12, "title": 20, "near_duplicate": 24}``. The keys are
    #: the source-level guard plus :attr:`pipeline.dedup.DedupeResult.by_tier`.
    duplicates_by_stage: dict[str, int] = Field(default_factory=dict)
    window_days: int = 30
    offline: bool = False
    generated_at: datetime | None = None
    source_errors: dict[str, str] = Field(default_factory=dict)


class IndexStats(_Model):
    """What the vector store currently holds.

    ``per_kind`` counts **documents**, matching
    :attr:`IngestStats.per_kind`, so ``/stats`` and the ingest report can be read
    side by side. Counting chunks here instead would report 259 policy items for a
    corpus of 200 policy documents, and the two numbers appearing under the same
    name is a trap for whoever reads the dashboards.
    """

    documents: int = 0
    chunks: int = 0
    dimensions: int = 0
    embedder: str = ""
    backend: str = ""
    #: Documents per source family: ``{"news": 200, "policy": 200, "price": 305}``.
    per_kind: dict[str, int] = Field(default_factory=dict)
    earliest: datetime | None = None
    latest: datetime | None = None
