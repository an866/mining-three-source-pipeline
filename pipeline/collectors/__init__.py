"""The three source adapters, behind one factory.

``build_collectors`` is the only constructor the rest of the project uses, so the
set of sources is declared in exactly one place and the API's ``/stats`` endpoint
cannot drift from what ingest actually collected.
"""

from __future__ import annotations

from pipeline.collectors.base import Collector, SourceReport, documents_from_records, in_window
from pipeline.collectors.news import FEEDS, NewsCollector
from pipeline.collectors.policy import POLICY_SOURCES, PolicyCollector, PolicySource
from pipeline.collectors.price import PRICE_ENDPOINTS, PriceCollector
from pipeline.config import Settings, get_settings

__all__ = [
    "FEEDS",
    "POLICY_SOURCES",
    "PRICE_ENDPOINTS",
    "Collector",
    "NewsCollector",
    "PolicyCollector",
    "PolicySource",
    "PriceCollector",
    "SourceReport",
    "build_collectors",
    "documents_from_records",
    "in_window",
]


def build_collectors(settings: Settings | None = None) -> list[Collector]:
    """The three collectors, in the order they are reported.

    News, then policy, then prices: stable ordering matters because ingest
    attributes a duplicate to whichever source saw it first, and a run-to-run
    change in that attribution would make ``IngestStats`` non-comparable.
    """
    resolved = settings or get_settings()
    return [
        NewsCollector(settings=resolved),
        PolicyCollector(settings=resolved),
        PriceCollector(settings=resolved),
    ]
