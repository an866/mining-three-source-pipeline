"""Three-tier de-duplication cascade.

Mining news is syndicated aggressively. The same story appears on mining.com,
inside a Google News cluster, in a company release and again on a trade site --
each with its own URL, often with its own headline. A URL set alone therefore
catches almost none of the duplication that matters, and every duplicate that
survives dilutes recall@5: five slots spent on one story instead of five.

The cascade, cheapest tier first:

======== ============================================ ===========================
Tier     Key                                          Catches
======== ============================================ ===========================
1        ``Document.id`` (canonical URL + title + date) exact re-collection
2        normalised title, within one source family    cross-domain syndication
3        5-shingle Jaccard >= 0.90, via LSH banding    rewritten headline
======== ============================================ ===========================

Tier 1 overlaps the collector's own identity guard
(:meth:`pipeline.collectors.base.Collector.dedupe_documents`), which uses the same
key for a different reason -- keeping the per-source count honest before the floor
check. Whichever runs first takes the removal, and the collectors run first, so on
the offline corpus ``exact_id`` reports **0** and the injected re-collected copies
show up under ``collector_identity`` in ``IngestStats.duplicates_by_stage``. Tier 1
is kept as the guard for any caller that assembles documents without going through
a collector, and is unit-tested directly rather than through the corpus.

The 0.90 threshold is the one number here that had to be measured rather than
chosen, and it was measured twice. On the generated corpus, the most similar pair of
genuinely *different* documents sits at 0.86 (two long policy notices sharing a
clause), while a reprint under a rewritten headline sits at 1.00, because an
aggregator rewrites the headline and republishes the wire copy verbatim. The
threshold therefore has a real margin on both sides. The first measurement used a
0.85 threshold and it produced exactly one false merge -- which is what a
measurement is for. Details are in DATA_NOTES.md.

Tier 3 is the one that costs something, so it is guarded by LSH banding: 64
shingles split into 16 bands of 4, documents only compared when they collide in
at least one band. That turns an O(n^2) pairwise comparison into one that, at
this corpus size, computes a few hundred candidate pairs instead of a few
thousand -- and stays correct if the corpus grows by two orders of magnitude.

Tiers 1 and 2 are scoped to the whole corpus; tier 3 is scoped *within a source
family*. A policy notice and a news item can legitimately share a sentence or two
of boilerplate, and merging them would delete a primary source in favour of a
secondary one.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from pipeline.logging_setup import get_logger
from pipeline.models import Document, SourceKind
from pipeline.textutils import dedupe_fingerprint, jaccard, normalize_title

__all__ = ["DedupeResult", "Deduper", "dedupe_documents"]

_log = get_logger(__name__)

#: LSH bands for the tier-3 pre-filter. 16 bands of 4 shingles gives an S-curve that
#: starts admitting pairs well below the decision threshold, so the filter never
#: rejects a true duplicate. Reducing the candidate set this way is what keeps tier
#: 3 off the O(n^2) path: on a 730-document corpus it compares a few hundred
#: candidate pairs instead of a quarter of a million.
_BANDS = 16
_BAND_SIZE = 4

#: A duplicate only displaces its incumbent when it is at least this much longer.
#: See :meth:`Deduper._reconcile` -- the band is what stops a byline-only
#: syndication from outranking the primary publication.
_MATERIAL_GROWTH = 1.2


def _bands(fingerprint: frozenset[str]) -> list[tuple[int, str]]:
    """``(band index, band key)`` pairs, spread across the whole signature.

    Bands are strided rather than taken from the prefix of the signature. A prefix
    would only ever hash the same 64 shingles, so a document whose edit touched a
    low-sorted shingle could fail to collide with its own reprint and be missed
    entirely -- the pre-filter would silently become a false-negative source
    instead of a performance optimisation.
    """
    if len(fingerprint) < _BAND_SIZE:
        return []
    ordered = sorted(fingerprint)
    stride = max(_BAND_SIZE, len(ordered) // _BANDS)
    pairs: list[tuple[int, str]] = []
    for band_index in range(_BANDS):
        start = band_index * stride
        band = ordered[start : start + _BAND_SIZE]
        if len(band) < _BAND_SIZE:
            break
        pairs.append((band_index, "|".join(band)))
    return pairs


@dataclass(slots=True)
class DedupeResult:
    """What the cascade did, per tier, so the run report can show its work."""

    documents: list[Document] = field(default_factory=list)
    by_tier: dict[str, int] = field(default_factory=dict)
    candidates_compared: int = 0

    @property
    def removed(self) -> int:
        """Total documents dropped."""
        return sum(self.by_tier.values())


class Deduper:
    """Stateful de-duplicator; one instance per pipeline run.

    Stateful rather than a pure function because tier 3 has to be able to *revise*
    an earlier decision: if a fuller copy of a story arrives after a shorter one,
    the survivor should be the fuller copy, and the shorter one already in the
    output list has to be replaced.
    """

    def __init__(
        self,
        *,
        near_duplicate_threshold: float = 0.9,
        max_candidates: int = 40,
    ) -> None:
        self.threshold = near_duplicate_threshold
        #: Safety valve: a pathological band (e.g. 200 wire stories sharing a
        #: lead sentence) must not turn into 200 comparisons for the 201st item.
        self.max_candidates = max_candidates
        self._by_id: dict[str, int] = {}
        self._by_title: dict[str, int] = {}
        self._by_band: dict[tuple[SourceKind, int, str], list[int]] = defaultdict(list)
        self._fingerprints: list[frozenset[str]] = []
        self._output: list[Document] = []
        self._tier_counts: dict[str, int] = defaultdict(int)
        self._compared = 0

    # -- public API ---------------------------------------------------------

    def run(self, documents: Iterable[Document]) -> DedupeResult:
        for document in documents:
            self.add(document)
        return DedupeResult(
            documents=list(self._output),
            by_tier=dict(self._tier_counts),
            candidates_compared=self._compared,
        )

    def add(self, document: Document) -> Document | None:
        """Insert one document. Returns it, or ``None`` when it was dropped.

        Returning the survivor-or-``None`` rather than a bool keeps the caller
        honest: there is no "which one won" ambiguity to get wrong.
        """
        if document.id in self._by_id:
            self._tier_counts["exact_id"] += 1
            return None

        title_key = f"{document.kind.value}|{normalize_title(document.title)}"
        if title_key in self._by_title:
            index = self._by_title[title_key]
            return self._reconcile(index, document, tier="title")

        fingerprint = dedupe_fingerprint(document.text)
        duplicate = self._find_near_duplicate(document, fingerprint)
        if duplicate is not None:
            return self._reconcile(duplicate, document, tier="near_duplicate")

        self._insert(document, fingerprint)
        return document

    # -- internals ----------------------------------------------------------

    def _find_near_duplicate(self, document: Document, fingerprint: frozenset[str]) -> int | None:
        """Index lookup on LSH bands, then a real Jaccard confirmation."""
        candidates: set[int] = set()
        for band_index, band_key in _bands(fingerprint):
            key = (document.kind, band_index, band_key)
            for index in self._by_band.get(key, ()):
                candidates.add(index)
                if len(candidates) >= self.max_candidates:
                    break
        if not candidates:
            return None

        best: tuple[float, int] | None = None
        for index in sorted(candidates):
            self._compared += 1
            similarity = jaccard(fingerprint, self._fingerprints[index])
            if similarity < self.threshold:
                continue
            if best is None or similarity > best[0]:
                best = (similarity, index)
        return best[1] if best else None

    def _reconcile(
        self, existing_index: int, challenger: Document, *, tier: str
    ) -> Document | None:
        """Two documents are the same story -- keep the better copy.

        "Better" means *materially* longer: a fuller copy is more retrievable and
        produces better citations. The 20% band is the important part. Without it,
        a syndicated copy that merely appends a byline would displace the primary
        publication, and every citation would point at an aggregator instead of the
        original. A materially longer body, on the other hand, is a genuinely
        different copy -- a truncated feed summary versus the article itself -- and
        should win.

        Ties go to the incumbent, which keeps the outcome independent of collection
        order among equal copies.
        """
        incumbent = self._output[existing_index]
        if len(challenger.text) > len(incumbent.text) * _MATERIAL_GROWTH:
            self._replace(existing_index, challenger)
            self._tier_counts[tier] += 1
            _log.debug(
                "duplicate replaced by a materially fuller copy",
                extra={
                    "extra_fields": {
                        "tier": tier,
                        "kept": challenger.id,
                        "dropped": incumbent.id,
                        "growth": round(len(challenger.text) / len(incumbent.text), 3),
                    }
                },
            )
            return challenger
        self._tier_counts[tier] += 1
        return None

    def _replace(self, index: int, document: Document) -> None:
        old = self._output[index]
        self._by_id.pop(old.id, None)
        self._by_title.pop(f"{old.kind.value}|{normalize_title(old.title)}", None)
        # Bands are left pointing at ``index``: they are a candidate generator,
        # and the fingerprint at that slot is about to be overwritten with the
        # near-identical new one. Rebuilding them would cost more than the stale
        # candidates cost.
        self._output[index] = document
        self._fingerprints[index] = dedupe_fingerprint(document.text)
        self._by_id[document.id] = index
        self._by_title[f"{document.kind.value}|{normalize_title(document.title)}"] = index

    def _insert(self, document: Document, fingerprint: frozenset[str]) -> None:
        index = len(self._output)
        self._output.append(document)
        self._fingerprints.append(fingerprint)
        self._by_id[document.id] = index
        self._by_title[f"{document.kind.value}|{normalize_title(document.title)}"] = index
        for band_index, band_key in _bands(fingerprint):
            self._by_band[(document.kind, band_index, band_key)].append(index)


def dedupe_documents(documents: Sequence[Document], *, threshold: float = 0.9) -> DedupeResult:
    """Convenience wrapper for a one-shot de-duplication pass."""
    return Deduper(near_duplicate_threshold=threshold).run(documents)
