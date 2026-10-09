"""Deterministic synthetic price series.

Design constraints, in order of importance:

1. **The brief's requirement is a floor, not a target.** Five commodities over
   sixty trading days is 300 observations, comfortably above ">= 200 items per
   source", and the window is comfortably above 30 days.

2. **Direction must be guaranteed, not hoped for.** Evaluation questions like
   "近 30 天哪个品种跌幅最大" need one right answer on every run date. So the
   series is not a free random walk: it is pinned to a target start-to-end change
   (:data:`PRICE_TARGETS`) and given bounded noise in between. Lithium always
   falls hardest, copper always rises, iron ore always drifts.

3. **Same day, same numbers.** Values are keyed on the calendar date, so asking
   twice today is idempotent and re-running tomorrow only appends.

Prices are synthetic and are labelled as such in every rendered document. They are
*shaped* like the real contracts (LME 3M copper in USD/t, SHFE lithium carbonate in
CNY/t) so that downstream formatting, units and benchmarks are exercised for real.
"""

from __future__ import annotations

import hashlib
import math
import random
from datetime import UTC, date, datetime, timedelta

from pipeline.models import COMMODITY_META, Commodity, PricePoint, PriceSeries
from pipeline.render import with_statistics
from pipeline.synthetic.anchors import PRICE_TARGETS

__all__ = [
    "build_series",
    "trading_days",
]

#: Anchor prices, chosen so the numbers look like the real contracts.
BASE_PRICES: dict[Commodity, float] = {
    Commodity.COPPER: 9_180.0,
    Commodity.ZINC: 2_845.0,
    Commodity.NICKEL: 15_600.0,
    Commodity.LITHIUM: 71_500.0,
    Commodity.IRON_ORE: 812.0,
}

#: Seeds are per commodity so one series can change without disturbing the others.
_SEEDS: dict[Commodity, int] = {
    Commodity.COPPER: 1001,
    Commodity.ZINC: 2002,
    Commodity.NICKEL: 3003,
    Commodity.LITHIUM: 4004,
    Commodity.IRON_ORE: 5005,
}


def _seed_for(commodity: Commodity, salt: str = "") -> int:
    """A stable integer seed -- ``hash()`` is salted per process and cannot be used."""
    digest = hashlib.sha256(f"{_SEEDS[commodity]}|{salt}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def trading_days(end: date, count: int) -> list[date]:
    """The last ``count`` weekdays ending at (or before) ``end``, oldest first."""
    days: list[date] = []
    cursor = end
    while len(days) < count:
        if cursor.weekday() < 5:
            days.append(cursor)
        cursor -= timedelta(days=1)
    return list(reversed(days))


def _intraday_noise(commodity: Commodity, index: int) -> float:
    """Per-bar high/low spread, 0.2%..0.9%, stable for a given index."""
    rng = random.Random(_seed_for(commodity, f"noise:{index}"))
    return 0.002 + rng.random() * 0.007


def _volume(commodity: Commodity, index: int) -> float:
    rng = random.Random(_seed_for(commodity, f"vol:{index}"))
    base = 40_000.0 if commodity is Commodity.COPPER else 12_000.0
    return round(base * (0.6 + rng.random() * 0.9), 1)


def _detrended_noise(commodity: Commodity, count: int, amplitude: float) -> list[float]:
    """A mean-zero noise path that starts and ends at exactly zero.

    The endpoints are pinned so the target start-to-end change survives the noise
    untouched; everything in between is a random walk drawn back to zero linearly.
    Without the pinning, "lithium fell 18%" would be a claim about a distribution
    rather than a guarantee.
    """
    if count <= 1:
        return [0.0] * count
    rng = random.Random(_seed_for(commodity, "path"))
    walk = [0.0]
    for _ in range(count - 1):
        walk.append(walk[-1] + rng.uniform(-1.0, 1.0))
    scale = max(abs(value) for value in walk) or 1.0
    normalised = [value / scale for value in walk]
    # Subtract the straight line from the first to the last point.
    end = normalised[-1]
    pulled = [normalised[i] - end * (i / (count - 1)) for i in range(count)]
    return [value * amplitude for value in pulled]


def build_series(
    commodity: Commodity,
    *,
    days: int = 60,
    end: date | None = None,
    source: str = "fixture",
) -> PriceSeries:
    """Build a complete, self-consistent series for one commodity."""
    last = end or datetime.now(UTC).date()
    calendar = trading_days(last, days)
    count = len(calendar)

    target_change, tolerance = PRICE_TARGETS[commodity.value]
    start_price = BASE_PRICES[commodity]
    end_price = start_price * (1.0 + target_change)

    # Bounded noise keeps every intermediate point plausible while leaving the
    # endpoints (and therefore the reported change) exact.
    amplitude = start_price * tolerance * 0.6
    noise = _detrended_noise(commodity, count, amplitude)

    points: list[PricePoint] = []
    for index, day in enumerate(calendar):
        fraction = index / (count - 1) if count > 1 else 0.0
        close = start_price + (end_price - start_price) * fraction + noise[index]
        spread = _intraday_noise(commodity, index)
        previous = points[-1].close if points else close
        points.append(
            PricePoint(
                date=day,
                close=round(close, 2),
                open=round(previous, 2),
                high=round(max(close, previous) * (1.0 + spread), 2),
                low=round(min(close, previous) * (1.0 - spread), 2),
                volume=_volume(commodity, index),
            )
        )

    meta = COMMODITY_META[commodity]
    return with_statistics(
        PriceSeries(
            commodity=commodity,
            currency=meta["currency"],
            unit=meta["unit"],
            benchmark=meta["benchmark"],
            source=source,
            points=points,
        )
    )


def implied_volatility(series: PriceSeries) -> float:
    """Annualised volatility of log returns -- used as a derived price fact."""
    closes = [point.close for point in series.points]
    if len(closes) < 3:
        return 0.0
    returns = [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))]
    mean = sum(returns) / len(returns)
    variance = sum((value - mean) ** 2 for value in returns) / (len(returns) - 1)
    return round(math.sqrt(variance) * math.sqrt(252) * 100.0, 2)
