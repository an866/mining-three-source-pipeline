"""Injected syndication -- the duplicates the cascade is supposed to catch.

A de-duplication cascade that never removes anything is untested code. This module
deliberately plants the three duplicate patterns that mining news actually
exhibits, so that ``IngestStats.duplicates_removed`` is a real measurement rather
than a number that is always zero.

=====================  ==========================================================
Pattern                What it looks like in the wild
=====================  ==========================================================
``exact``              the same URL re-collected on the next run (a feed that
                       keeps 40 items online, a retry after a timeout)
``same_title``         one wire story republished by three outlets under the
                       identical headline
``rewritten``          the same story under a rewritten headline -- "raises"
                       becomes "lifts", an outlet adds its own attribution
=====================  ==========================================================

Where each pattern is caught: the ``exact`` copies are removed by the source-level
identity guard in the collector (same key as the cascade's tier 1, and the guard
runs first), ``same_title`` by tier 2, and ``rewritten`` by tier 3. All three are
accounted for in ``IngestStats.duplicates_by_stage`` -- see the note in
``pipeline/dedup.py``.

Reprints are generated only from *filler* documents. The anchored documents carry
the evaluation set's ground truth, and a reprint that displaced one -- the cascade
keeps the longer copy -- would silently break recall@5 by changing which URL holds
the fact. Keeping the anchors clean makes the eval independent of the dedup
behaviour being measured.

No reprint is longer than its original, so the *original* survives reconciliation
and a reader who follows a citation always lands on the primary publication.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

__all__ = ["SYNDICATED_OUTLETS", "syndicate"]

#: Second-tier outlets that reprint wire copy. Deliberately distinct from the
#: primary outlets in :mod:`pipeline.synthetic.text_corpus` so a reprint is never
#: confused with an original collection of the same feed.
SYNDICATED_OUTLETS: tuple[str, ...] = (
    "metals-daily",
    "resourcewire",
    "globalminingreview",
    "miningbulletin",
)

#: How many reprints each style contributes across the corpus, per source.
_EXACT_COPIES = 6
_TITLE_COPIES = 10
_REWRITE_COPIES = 12

#: Headline rewrites, applied to the *first word* of the title where one matches.
#: These are the substitutions wire desks actually make to avoid duplicate
#: headlines in an aggregator.
_SYNONYMS: tuple[tuple[str, str], ...] = (
    ("raises", "lifts"),
    ("rise", "climb"),
    ("rises", "climbs"),
    ("falls", "slips"),
    ("fall", "slip"),
    ("cuts", "trims"),
    ("lowers", "reduces"),
    ("agrees", "agrees to"),
    ("announces", "announces new"),
    ("reports", "flags"),
)


def syndicate(
    records: Iterable[Mapping[str, Any]],
    *,
    source_outlets: tuple[str, ...] = SYNDICATED_OUTLETS,
) -> list[dict[str, Any]]:
    """Return reprint records for ``records``.

    The caller appends the result to the original list; nothing is removed here,
    because de-duplication is the pipeline's job, not the generator's. Making the
    generator do the de-duplication would test the generator instead of the cascade.
    """
    originals = list(records)
    if not originals:
        return []

    reprints: list[dict[str, Any]] = []
    reprints.extend(_exact_copies(originals))
    reprints.extend(_title_copies(originals, source_outlets))
    reprints.extend(_rewrite_copies(originals, source_outlets))
    return reprints


def _exact_copies(
    originals: list[Mapping[str, Any]], count: int = _EXACT_COPIES
) -> list[dict[str, Any]]:
    """Same URL, same title, same date -- the re-collection case.

    Tagged ``recollected`` purely so tests and the corpus probe can tell an injected
    duplicate from an original; tags are not part of a document's identity, so the
    tag does not affect which tier catches it.
    """
    copies: list[dict[str, Any]] = []
    for record in _spread_pick(originals, count):
        copy = dict(record)
        copy["tags"] = [*_tags(record), "recollected"]
        copies.append(copy)
    return copies


def _title_copies(
    originals: list[Mapping[str, Any]], outlets: tuple[str, ...], count: int = _TITLE_COPIES
) -> list[dict[str, Any]]:
    """Same headline, different outlet and URL -- the syndication case."""
    copies: list[dict[str, Any]] = []
    for offset, record in enumerate(_spread_pick(originals, count)):
        outlet = outlets[offset % len(outlets)]
        copy = dict(record)
        copy["url"] = _reprint_url(str(record.get("url") or ""), outlet, offset, tag="synd")
        copy["source"] = outlet
        copy["tags"] = [*_tags(record), "syndicated"]
        copies.append(copy)
    return copies


def _rewrite_copies(
    originals: list[Mapping[str, Any]], outlets: tuple[str, ...], count: int = _REWRITE_COPIES
) -> list[dict[str, Any]]:
    """Rewritten headline, unchanged body -- the case tiers 1 and 2 cannot see.

    The body is deliberately left byte-identical, because that is what an
    aggregator actually does: it rewrites the headline and republishes the wire
    copy verbatim. It is also the only construction that makes tier 3 *necessary* --
    the primary key differs (the title is in it), and the title key differs by
    construction, so content similarity is the only signal left.

    An earlier version also mutated the final sentence. That was removed for two
    reasons: it made the reprint longer than its original (so the cascade kept the
    reprint instead of the primary publication), and on CJK text the one-character
    edit destroyed a disproportionate share of 5-token shingles, dropping similarity
    below any threshold that would not also merge genuinely distinct documents.
    """
    copies: list[dict[str, Any]] = []
    for offset, record in enumerate(_spread_pick(originals, count)):
        outlet = outlets[(offset + 1) % len(outlets)]
        copy = dict(record)
        copy["title"] = _rewrite_title(str(record.get("title") or ""))
        copy["url"] = _reprint_url(str(record.get("url") or ""), outlet, offset, tag="repub")
        copy["source"] = outlet
        copy["tags"] = [*_tags(record), "reprint"]
        copies.append(copy)
    return copies


def _rewrite_title(title: str) -> str:
    """Rewrite a headline the way a wire desk would.

    Two substitutions, in order of realism: swap the leading verb for a synonym
    ("raises" -> "lifts"), or -- when no verb matches -- prefix a marker.

    The marker goes at the **front**, never the end. ``normalize_title`` strips a
    trailing ``" - publisher"`` suffix, so appending " (analysis)" to a headline
    like "X - mining.com" normalises straight back to "x" and the rewrite would be
    silently identical to its original for de-duplication purposes. That bug was
    caught by measuring, not by reading.
    """
    words = title.split()
    for index, word in enumerate(words[:6]):
        bare = word.strip(",.:;()[]")
        for needle, replacement in _SYNONYMS:
            if bare.lower() == needle:
                words[index] = word.replace(bare, replacement)
                return " ".join(words)
    # Chinese headlines do not capitalise or inflect; outlets mark a reprint with a
    # banner instead.
    return f"【转载】{title}" if _has_cjk(title) else f"Analysis: {title}"


def _reprint_url(url: str, outlet: str, offset: int, *, tag: str) -> str:
    """A distinct URL on the reprinting outlet's domain."""
    slug = url.split("//", 1)[-1]
    return f"https://www.{outlet}.example/{tag}/{offset}-{slug.replace('/', '-')[:120]}"


def _tags(record: Mapping[str, Any]) -> list[str]:
    tags = record.get("tags")
    return [str(tag) for tag in tags] if isinstance(tags, list | tuple) else []


def _spread_pick(records: list[Mapping[str, Any]], count: int) -> list[Mapping[str, Any]]:
    """Evenly spaced, deterministic selection rather than the first ``count`` items.

    Picking a prefix would make every duplicate a duplicate of the same handful of
    documents; spreading the selection exercises reconciliation across the whole
    corpus, including the case where a reprint arrives long after its original.
    """
    if count <= 0 or not records:
        return []
    if count >= len(records):
        return list(records)
    step = len(records) / count
    return [records[min(len(records) - 1, int(index * step))] for index in range(count)]


def _has_cjk(text: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in text)
