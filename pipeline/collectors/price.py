"""Source 3 -- commodity prices.

Five benchmarks, three venues:

===============  =======================  ========
Commodity        Benchmark                Venue
===============  =======================  ========
copper           LME 3M Copper            LME
zinc             LME 3M Zinc              LME
nickel           LME 3M Nickel            LME
lithium          SHFE Lithium Carbonate   SHFE
iron ore         Mysteel 62% Fe CFR       Mysteel
===============  =======================  ========

The interesting part is not fetching numbers, it is making them *retrievable*.
A vector store cannot embed ``9150.5``, so every observation is also rendered as
a bilingual market note ("碳酸锂 2026-08-12 收盘 71,500.00 CNY/t … 近 30 个交易日
下跌 18.50%") by :mod:`pipeline.render`. That rendering is what lets a question
like "近一个月锂价怎么走" hit a numeric table.

The live path accepts three payload shapes because the three venues disagree:
LME-style JSON, SHFE-style CSV, and Mysteel-style HTML tables. All three are
normalised into :class:`~pipeline.models.PriceSeries` *before* anything downstream
touches them, so the renderer, the vector store and the API only ever see one
shape.
"""

from __future__ import annotations

import csv
import io
import json
import re
from datetime import UTC, date, datetime, timedelta
from typing import Any

from pipeline.cache import DiskCache
from pipeline.collectors.base import Collector, documents_from_records
from pipeline.config import Settings
from pipeline.errors import ParseError, PipelineError
from pipeline.http import HttpClient
from pipeline.logging_setup import get_logger
from pipeline.models import (
    COMMODITY_META,
    Commodity,
    Document,
    PricePoint,
    PriceSeries,
    SourceKind,
)
from pipeline.normalize import records_from_price_series
from pipeline.render import with_statistics
from pipeline.synthetic.prices import build_series
from pipeline.textutils import parse_datetime

__all__ = ["PRICE_ENDPOINTS", "PRICE_HISTORY_DAYS", "PRICE_SESSIONS", "PriceCollector"]

_log = get_logger(__name__)

#: How many sessions of history the price source keeps. Sixty sessions is about
#: three months of weekdays -- enough for the trend and range figures a market
#: question actually turns on, and safely inside the 91-day collection horizon
#: below, so nothing is dropped by the window filter.
PRICE_SESSIONS = 60

#: The collection horizon for prices, in calendar days. 91 days is 13 weeks, which
#: comfortably contains 60 weekdays once public holidays are ignored.
PRICE_HISTORY_DAYS = 91


#: ``commodity -> (venue, endpoint template, payload shape)``. The ``{start}`` and
#: ``{end}`` placeholders are ISO dates.
PRICE_ENDPOINTS: dict[Commodity, tuple[str, str, str]] = {
    Commodity.COPPER: (
        "LME",
        "https://www.lme.com/api/prices/copper?from={start}&to={end}",
        "lme-json",
    ),
    Commodity.ZINC: (
        "LME",
        "https://www.lme.com/api/prices/zinc?from={start}&to={end}",
        "lme-json",
    ),
    Commodity.NICKEL: (
        "LME",
        "https://www.lme.com/api/prices/nickel?from={start}&to={end}",
        "lme-json",
    ),
    Commodity.LITHIUM: (
        "SHFE",
        "https://www.shfe.com.cn/data/dailydata/kx/kx{yyyymmdd}.dat",
        "shfe-csv",
    ),
    Commodity.IRON_ORE: (
        "Mysteel",
        "https://index.mysteel.com/price/getChartMultiCity_1_0.ms?code=62Fe",
        "mysteel-html",
    ),
}


class PriceCollector(Collector):
    """Collects OHLC series for the five benchmark commodities."""

    name = "price"
    kind = SourceKind.PRICE
    history_days = PRICE_HISTORY_DAYS

    def __init__(
        self,
        *,
        commodities: tuple[Commodity, ...] | None = None,
        sessions: int = PRICE_SESSIONS,
        settings: Settings | None = None,
        cache: DiskCache | None = None,
        client: HttpClient | None = None,
    ) -> None:
        super().__init__(settings=settings, cache=cache, client=client)
        self.commodities = commodities or tuple(Commodity)
        self.sessions = sessions

    # -- public API ---------------------------------------------------------

    def collect(self, *, days: int) -> list[Document]:
        """One document per commodity per session, plus a window summary each.

        Unlike the other two sources, price data is *always* available: the
        fixture is the source of record here. The brief's five benchmarks are
        licensed feeds, and no public endpoint returns a consistent 60-session
        history for all five. So the live path is attempted and its result is
        used when it parses, but the fallback is not an error state -- it is the
        expected outcome, and the documents say so.
        """
        return self._build(days=days, end=None)

    def collect_fixture(self, *, days: int, end: date | None = None) -> list[Document]:
        """Force the deterministic corpus, bypassing any network attempt."""
        return self._build(days=days, end=end, force_fixture=True)

    # -- internals ----------------------------------------------------------

    def _build(self, *, days: int, end: date | None, force_fixture: bool = False) -> list[Document]:
        # ``days`` is the collection window the base class already widened to
        # ``history_days``; the *series length* stays fixed at ``self.sessions`` so
        # the summary figures ("近 60 个交易日") mean the same thing regardless of
        # how the window was requested.
        sessions = self.sessions
        documents: list[Document] = []
        for commodity in self.commodities:
            series = (
                build_series(commodity, days=sessions, end=end, source="fixture")
                if force_fixture or self._settings.offline
                else self._series_for(commodity, days=days, sessions=sessions)
            )
            records = records_from_price_series(series, daily_days=sessions)
            documents.extend(documents_from_records(records, source=self.name))
        return self.dedupe_documents(documents)

    def _series_for(self, commodity: Commodity, *, days: int, sessions: int) -> PriceSeries:
        """Live series when it parses, fixture otherwise. Never raises."""
        try:
            series = self._fetch_series(commodity, days=days, sessions=sessions)
        except PipelineError as exc:
            _log.info(
                "price endpoint unusable; using deterministic fixture",
                extra={"extra_fields": {"commodity": commodity.value, "error": str(exc)}},
            )
            return build_series(commodity, days=sessions, source="fixture")
        if series is None or len(series.points) < 2:
            return build_series(commodity, days=sessions, source="fixture")
        return series

    def _fetch_series(
        self, commodity: Commodity, *, days: int, sessions: int
    ) -> PriceSeries | None:
        venue, template, shape = PRICE_ENDPOINTS[commodity]
        end = datetime.now(UTC).date()
        # Calendar headroom: a 30-session window spans ~45 calendar days.
        start = end - timedelta(days=days * 2)
        url = template.format(
            start=start.isoformat(),
            end=end.isoformat(),
            yyyymmdd=end.strftime("%Y%m%d"),
        )
        cached = self._cache.get("prices", url)
        if cached is None:
            payload = self.client.get_text(url)
            self._cache.set("prices", url, payload)
        else:
            payload = str(cached)

        meta = COMMODITY_META[commodity]
        points = _parse_payload(payload, shape=shape, what=f"{venue}:{commodity.value}")
        if not points:
            raise ParseError(f"{venue} payload for {commodity.value}", "no usable observations")
        return with_statistics(
            PriceSeries(
                commodity=commodity,
                currency=meta["currency"],
                unit=meta["unit"],
                benchmark=meta["benchmark"],
                source=venue,
                points=points[-sessions:],
            )
        )


# ---------------------------------------------------------------------------
# Payload adapters
# ---------------------------------------------------------------------------


def _parse_payload(payload: str, *, shape: str, what: str) -> list[PricePoint]:
    """Dispatch on the venue's payload shape."""
    if shape == "lme-json":
        return _parse_lme_json(payload, what=what)
    if shape == "shfe-csv":
        return _parse_shfe_csv(payload, what=what)
    if shape == "mysteel-html":
        return _parse_mysteel_table(payload, what=what)
    raise ParseError(what, f"unknown payload shape {shape!r}")


def _parse_lme_json(payload: str, *, what: str) -> list[PricePoint]:
    """LME publishes a JSON array of ``{date, open, high, low, close, volume}``."""
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ParseError(what, str(exc)) from exc
    rows: Any = data.get("prices", data) if isinstance(data, dict) else data
    if not isinstance(rows, list):
        raise ParseError(what, "expected a list of observations")
    points: list[PricePoint] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        when = parse_datetime(str(row.get("date", "")))
        close = _as_float(row.get("close"))
        if when is None or close is None:
            continue
        points.append(
            PricePoint(
                date=when.date(),
                close=close,
                open=_as_float(row.get("open")),
                high=_as_float(row.get("high")),
                low=_as_float(row.get("low")),
                volume=_as_float(row.get("volume")),
            )
        )
    return sorted(points, key=lambda point: point.date)


def _parse_shfe_csv(payload: str, *, what: str) -> list[PricePoint]:
    """SHFE's ``kx*.dat`` files are comma-separated with a Chinese header row."""
    reader = csv.DictReader(io.StringIO(payload.strip()))
    if reader.fieldnames is None:
        raise ParseError(what, "empty CSV")
    # Header names vary between deployments; match on what is present rather than
    # on exact strings.
    fields = {name.strip(): name for name in reader.fieldnames}

    def pick(*candidates: str) -> str | None:
        for candidate in candidates:
            if candidate in fields:
                return fields[candidate]
        return None

    date_key = pick("日期", "date", "trade_date")
    close_key = pick("收盘价", "close", "settlement", "settle")
    if date_key is None or close_key is None:
        raise ParseError(what, f"unrecognised CSV columns: {reader.fieldnames}")
    points: list[PricePoint] = []
    for row in reader:
        when = parse_datetime((row.get(date_key) or "").strip())
        close = _as_float(row.get(close_key))
        if when is None or close is None:
            continue
        points.append(
            PricePoint(
                date=when.date(),
                close=close,
                open=_as_float(row.get(pick("开盘价", "open") or "")),
                high=_as_float(row.get(pick("最高价", "high") or "")),
                low=_as_float(row.get(pick("最低价", "low") or "")),
                volume=_as_float(row.get(pick("成交量", "volume") or "")),
            )
        )
    return sorted(points, key=lambda point: point.date)


_ROW_RE = re.compile(r"<tr[^>]*>(?P<cells>.*?)</tr>", re.IGNORECASE | re.DOTALL)
_CELL_RE = re.compile(r"<t[dh][^>]*>(?P<value>.*?)</t[dh]>", re.IGNORECASE | re.DOTALL)


def _parse_mysteel_table(payload: str, *, what: str) -> list[PricePoint]:
    """Mysteel index pages carry a plain HTML table of date/price rows."""
    points: list[PricePoint] = []
    for row in _ROW_RE.finditer(payload):
        cells = [
            re.sub(r"<[^>]+>", "", match.group("value")).strip()
            for match in _CELL_RE.finditer(row.group("cells"))
        ]
        if len(cells) < 2:
            continue
        when = parse_datetime(cells[0])
        close = _as_float(cells[1])
        if when is None or close is None:
            continue
        points.append(PricePoint(date=when.date(), close=close))
    if not points:
        # Same contract as the JSON and CSV adapters: an adapter that finds nothing
        # useful raises, and the caller degrades to the deterministic fixture. The
        # alternative -- returning an empty list -- looks like "the venue is quiet"
        # and silently produces a commodity with no price history at all.
        raise ParseError(what, "no usable observations in the HTML table")
    return sorted(points, key=lambda point: point.date)


def _as_float(value: object) -> float | None:
    """Tolerant numeric coercion: ``"1,234.5"`` and ``"—"`` both appear in the wild."""
    if value is None:
        return None
    if isinstance(value, int | float):
        return float(value)
    text = str(value).strip().replace(",", "").replace("\u00a0", "")
    if not text or text in {"-", "—", "–", "N/A", "null"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None
