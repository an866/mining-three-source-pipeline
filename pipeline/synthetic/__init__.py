"""Deterministic synthetic corpus, used when the real sources are unreachable.

The sandbox and any CI runner block the upstream sources this brief names
(mining.com answers 403 to non-browser agents, Google News blocks datacentre
IPs, stooq serves a JavaScript challenge, Eastmoney requires a ``Referer``). The
real adapters are implemented and are what ``MINING_OFFLINE=0`` uses; this package
is what makes the pipeline *demonstrable and measurable* on any machine.

Three properties are load-bearing:

* **deterministic** -- the same date yields the same corpus, so an evaluation
  score is reproducible rather than a snapshot of a random draw;
* **shaped like the real thing** -- anchors are modelled on real instruments and
  real market structure, prices on real contracts and units;
* **labelled** -- every synthetic document says so in ``DATA_NOTES.md`` and in the
  ``/stats`` response. Nothing here pretends to be collected data.
"""

from pipeline.synthetic.anchors import ANCHORS, PRICE_TARGETS, Anchor, anchor_urls
from pipeline.synthetic.prices import build_series
from pipeline.synthetic.syndication import SYNDICATED_OUTLETS, syndicate
from pipeline.synthetic.text_corpus import generate_news_records, generate_policy_records

__all__ = [
    "ANCHORS",
    "PRICE_TARGETS",
    "SYNDICATED_OUTLETS",
    "Anchor",
    "anchor_urls",
    "build_series",
    "generate_news_records",
    "generate_policy_records",
    "syndicate",
]
