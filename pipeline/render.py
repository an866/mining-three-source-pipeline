"""Deriving and rendering price series into embeddable prose.

Two jobs, one module, because they share a single source of truth about a series:

* :func:`with_statistics` computes the derived fields (start/end/change/range/
  direction) **once**, at construction time. The API, the evaluation harness and
  the rendered text all read those fields, so they cannot disagree -- a chart that
  says "lithium fell 18%" and an answer that says "lithium rose" is impossible by
  construction.
* :func:`render_day_document` / :func:`render_summary_document` turn the numbers
  into text. A vector store cannot embed ``71,500.00``; it *can* embed "碳酸锂
  2026-08-12 收盘 71,500.00 CNY/t，当日下跌 0.8%" -- and that is exactly what a
  question like "锂价最近怎么走" needs to retrieve.

Being the only place that produces this text means the live and offline paths
emit identically worded documents, so the evaluation set cannot be tuned to one
of them.
"""

from __future__ import annotations

from pipeline.models import COMMODITY_META, PricePoint, PriceSeries
from pipeline.textutils import normalize_whitespace

__all__ = [
    "direction_word",
    "render_day_document",
    "render_summary_document",
    "with_statistics",
]

#: Below this percentage change a series is described as flat rather than up or
#: down. Chosen over "exactly zero" because an 0.4% drift over a month is noise,
#: and calling it a direction would make the eval's direction questions unstable.
FLAT_THRESHOLD_PCT = 0.5


def with_statistics(series: PriceSeries) -> PriceSeries:
    """Populate the derived fields once, so every consumer agrees on them.

    Idempotent: recomputing from the same points yields the same values, so it is
    safe to call on a series that was already processed.
    """
    closes = [point.close for point in series.points]
    if not closes:
        return series
    start, end = closes[0], closes[-1]
    change_abs = end - start
    change_pct = (change_abs / start * 100.0) if start else 0.0
    series.start_price = round(start, 2)
    series.end_price = round(end, 2)
    series.change_abs = round(change_abs, 2)
    series.change_pct = round(change_pct, 2)
    series.min_price = round(min(closes), 2)
    series.max_price = round(max(closes), 2)
    if abs(change_pct) < FLAT_THRESHOLD_PCT:
        series.direction = "flat"
    else:
        series.direction = "up" if change_pct > 0 else "down"
    return series


def direction_word(direction: str, language: str = "zh") -> str:
    """Localised direction label."""
    if language == "zh":
        return {"up": "上涨", "down": "下跌", "flat": "基本持平"}.get(direction, "基本持平")
    return {"up": "rose", "down": "fell", "flat": "held steady"}.get(direction, "held steady")


def render_day_document(series: PriceSeries, point: PricePoint, *, index: int) -> str:
    """One trading day as a bilingual market note."""
    meta = COMMODITY_META[series.commodity]
    previous = series.points[index - 1].close if index > 0 else point.close
    day_change = (point.close - previous) / previous * 100.0 if previous else 0.0
    day_word_zh = "上涨" if day_change >= 0 else "下跌"
    day_word_en = "up" if day_change >= 0 else "down"

    zh = (
        f"{meta['zh']}（{series.commodity.value}）{point.date.isoformat()} 收盘 "
        f"{point.close:,.2f} {series.unit}，基准 {series.benchmark}，"
        f"当日{day_word_zh} {abs(day_change):.2f}%。"
        f"当日区间 {point.low:,.2f} - {point.high:,.2f} {series.unit}，"
        f"成交量 {point.volume:,.0f}。"
    )
    en = (
        f"{series.benchmark} settled at {point.close:,.2f} {series.unit} on "
        f"{point.date.isoformat()}, {day_word_en} {abs(day_change):.2f}% on the session. "
        f"Session range {point.low:,.2f}-{point.high:,.2f} {series.unit}."
    )
    trend = (
        f"Over the last {len(series.points)} sessions {series.commodity.value} "
        f"{direction_word(series.direction, 'en')} {abs(series.change_pct or 0.0):.2f}% "
        f"(from {series.start_price:,.2f} to {series.end_price:,.2f} {series.unit}), "
        f"range {series.min_price:,.2f}-{series.max_price:,.2f}. "
        f"{meta['zh']}近 {len(series.points)} 个交易日"
        f"{direction_word(series.direction)} {abs(series.change_pct or 0.0):.2f}%，"
        f"区间 {series.min_price:,.2f}-{series.max_price:,.2f} {series.unit}。"
    )
    return normalize_whitespace(f"{zh}\n{en}\n{trend}")


def render_summary_document(series: PriceSeries) -> str:
    """The window-level summary for one commodity.

    Leads with the change and the direction because those are what a natural
    language question about a price trend is actually asking for.
    """
    meta = COMMODITY_META[series.commodity]
    sessions = len(series.points)
    change = series.change_pct or 0.0
    return normalize_whitespace(
        f"{series.benchmark} ({meta['zh']}) {series.commodity.value} price trend over the last "
        f"{sessions} sessions: start {series.start_price:,.2f}, end {series.end_price:,.2f} "
        f"{series.unit}, change {change:+.2f}% ({series.change_abs:+,.2f} {series.unit}), "
        f"direction {series.direction}. Traded range {series.min_price:,.2f}-"
        f"{series.max_price:,.2f} {series.unit}. "
        f"近 {sessions} 个交易日{meta['zh']}价格{direction_word(series.direction)} "
        f"{abs(change):.2f}%，区间 {series.min_price:,.2f}-{series.max_price:,.2f} "
        f"{series.unit}，基准 {series.benchmark}。"
    )
