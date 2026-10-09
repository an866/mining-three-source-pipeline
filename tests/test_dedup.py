"""The three-tier de-duplication cascade.

Each tier is tested with a pair that only that tier can catch, because a cascade is
easy to get wrong in a way that still removes duplicates: the expensive tier can
cover for the cheap one and the cheap one is then dead code nobody notices.
"""

from __future__ import annotations

from tests.conftest import make_document

from pipeline.dedup import Deduper, dedupe_documents
from pipeline.models import Document, SourceKind
from pipeline.textutils import document_id, strip_tracking

_BODY = (
    "Pilbara Minerals raised full-year spodumene concentrate guidance to between "
    "800,000 and 840,000 dry metric tonnes, up from 760,000 to 800,000, after "
    "throughput work at the Pilgan plant lifted recoveries above design rates."
)


def _doc(doc_id: str, *, title: str, body: str, url: str, source: str = "mining.com") -> Document:
    return make_document(
        doc_id=doc_id,
        title=title,
        text=body,
        url=url,
        source=source,
        kind=SourceKind.NEWS,
        commodity="lithium",
    )


class TestTier1ExactId:
    """Tier 1 keys on ``Document.id``, which is derived from the canonical URL.

    The canonicalisation itself lives in ``textutils.document_id`` (the collectors
    build ids with it), so it is tested there rather than through the deduper -- a
    fabricated ``id`` cannot exercise it.
    """

    def test_two_documents_with_the_same_id_collapse(self) -> None:
        a = _doc("same-id", title="Pilbara raises guidance", body=_BODY, url="https://a.test/1")
        b = _doc(
            "same-id",
            title="Different headline entirely",
            body="Short re-collection.",
            url="https://a.test/1",
        )
        result = Deduper().run([a, b])
        assert len(result.documents) == 1
        assert result.by_tier.get("exact_id", 0) == 1

    def test_tracking_parameters_do_not_create_a_new_id(self) -> None:
        plain = "https://www.mining.com/pilbara-guidance-2026/"
        tracked = f"{plain}?utm_source=newsletter&utm_medium=email&fbclid=abc"
        assert document_id("news", plain, "Pilbara raises guidance") == document_id(
            "news", tracked, "Pilbara raises guidance"
        )
        assert strip_tracking(tracked) == plain

    def test_different_urls_are_kept(self) -> None:
        a = _doc("a", title="Pilbara raises guidance", body=_BODY, url="https://a.test/1")
        b = _doc("b", title="Zinc TC collapse", body="Zinc charges fell.", url="https://b.test/2")
        assert len(Deduper().run([a, b]).documents) == 2


class TestTier2NormalisedTitle:
    def test_same_headline_from_two_outlets_is_a_duplicate(self) -> None:
        a = _doc(
            "a",
            title="Indonesia trims nickel ore quota for 2027",
            body=_BODY,
            url="https://outlet-one.test/story",
            source="outlet-one",
        )
        b = _doc(
            "b",
            title="Indonesia trims nickel ore quota for 2027",
            body="A short wire version of the same story.",
            url="https://outlet-two.test/story",
            source="outlet-two",
        )
        result = Deduper().run([a, b])
        assert len(result.documents) == 1
        assert result.by_tier.get("title", 0) == 1

    def test_title_match_ignores_publisher_suffix(self) -> None:
        a = _doc(
            "a",
            title="Zinc treatment charges collapse",
            body="Zinc TC fell.",
            url="https://a.test/1",
            source="mining.com",
        )
        b = _doc(
            "b",
            title="Zinc treatment charges collapse - Reuters",
            body="Zinc TC fell sharply this week.",
            url="https://b.test/2",
            source="reuters",
        )
        assert len(Deduper().run([a, b]).documents) == 1


class TestTier3NearDuplicate:
    def test_rewritten_headline_same_body_is_a_duplicate(self) -> None:
        # Tier 2 cannot see this (different titles) and tier 1 cannot either
        # (different URLs); only shingle similarity catches it.
        a = _doc(
            "a",
            title="Pilbara Minerals raises spodumene guidance after Pilgan work",
            body=_BODY,
            url="https://a.test/1",
        )
        b = _doc(
            "b",
            title="Analysis: Pilbara Minerals raises spodumene guidance after Pilgan work",
            body=_BODY,
            url="https://a.test/1-reprint",
            source="metals-daily",
        )
        result = Deduper().run([a, b])
        assert len(result.documents) == 1
        assert result.by_tier.get("near_duplicate", 0) == 1

    def test_two_different_stories_about_one_topic_are_kept(self) -> None:
        # The threshold has to be high enough that "same topic" is not "duplicate".
        a = _doc("a", title="Pilbara raises spodumene guidance", body=_BODY, url="https://a.test/1")
        b = _doc(
            "b",
            title="Lithium destocking nears its end",
            body=(
                "Cathode makers reported utilisation rates recovering to the mid-70s "
                "percent and downstream inventories fell to about four weeks of "
                "consumption from nine weeks at the start of the year."
            ),
            url="https://a.test/2",
        )
        assert len(Deduper().run([a, b]).documents) == 2

    def test_threshold_is_configurable_and_monotonic(self) -> None:
        a = _doc("a", title="Pilbara raises guidance", body=_BODY, url="https://a.test/1")
        edited = _BODY.replace("800,000 and 840,000", "820,000 and 860,000").replace(
            "760,000 to 800,000", "780,000 to 820,000"
        )
        b = _doc("b", title="Pilbara upgrades guidance", body=edited, url="https://a.test/2")
        loose = dedupe_documents([a, b], threshold=0.5)
        strict = dedupe_documents([a, b], threshold=0.99)
        assert len(loose.documents) <= len(strict.documents)


class TestCascadeReporting:
    def test_by_tier_attributes_each_removal_to_one_tier(self) -> None:
        docs = [
            _doc("a", title="Same headline", body=_BODY, url="https://a.test/1"),
            _doc("b", title="Same headline", body=_BODY, url="https://a.test/1?utm_source=x"),
            _doc("c", title="Another headline", body=_BODY, url="https://a.test/2", source="other"),
        ]
        result = Deduper().run(docs)
        # At least two of the three collapse, and no tier double-counts a document.
        assert len(result.documents) <= 2
        assert sum(result.by_tier.values()) == 3 - len(result.documents)

    def test_removed_matches_the_arithmetic(self) -> None:
        docs = [
            _doc("a", title="Same", body=_BODY, url="https://a.test/1"),
            _doc("b", title="Same", body=_BODY, url="https://a.test/1?ref=x"),
        ]
        result = Deduper().run(docs)
        assert result.removed == 2 - len(result.documents)

    def test_empty_input_is_not_an_error(self) -> None:
        result = Deduper().run([])
        assert result.documents == []
        assert result.removed == 0

    def test_output_order_is_input_order(self) -> None:
        # The surviving representative must be deterministic, or re-running ingest
        # would produce a different set of documents from identical input.
        docs = [
            _doc("first", title="Alpha story", body=_BODY, url="https://a.test/1"),
            _doc(
                "second", title="Beta story", body="Unrelated copper news.", url="https://a.test/2"
            ),
        ]
        assert [d.id for d in Deduper().run(docs).documents] == ["first", "second"]
