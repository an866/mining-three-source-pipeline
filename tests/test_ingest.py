"""End-to-end offline ingest.

Marked ``slow``: this builds the whole corpus (a few seconds). It is the only test
that proves the brief's numbers -- >= 200 items per source, >= 30-day window,
>= 600 documents in the index -- and that the planted anchors survive de-duplication,
so it earns its runtime in CI rather than being mocked away.
"""

from __future__ import annotations

import json

import pytest

from pipeline.config import Settings
from pipeline.ingest import ingest, load_index
from pipeline.synthetic.anchors import ANCHORS

pytestmark = pytest.mark.slow

FLOOR_PER_SOURCE = 200
FLOOR_TOTAL = 600


@pytest.fixture(scope="module")
def outcome(tmp_path_factory: pytest.TempPathFactory):
    """One ingest for the whole module -- it is the expensive part."""
    directory = tmp_path_factory.mktemp("ingest")
    settings = Settings(
        index_dir=directory / "index",
        cache_dir=directory / "cache",
        cache_enabled=False,
        offline=True,
    )
    return settings, ingest(settings, days=30, index_dir=settings.index_dir)


class TestCorpusFloors:
    def test_each_source_clears_the_floor(self, outcome) -> None:
        _, result = outcome
        per_kind = result.stats.per_kind
        for kind in ("news", "policy", "price"):
            assert per_kind.get(kind, 0) >= FLOOR_PER_SOURCE, f"{kind} below the floor"

    def test_total_clears_the_floor(self, outcome) -> None:
        _, result = outcome
        assert result.stats.documents >= FLOOR_TOTAL

    def test_window_is_at_least_thirty_days(self, outcome) -> None:
        _, result = outcome
        assert result.stats.window_days >= 30

    def test_price_window_is_longer_than_the_news_window(self, outcome) -> None:
        # Prices are a time series: 30 documents would not support "the last 60
        # sessions", so the price collector declares its own history window.
        _, result = outcome
        assert result.stats.per_kind["price"] > FLOOR_PER_SOURCE

    def test_no_source_errors_offline(self, outcome) -> None:
        _, result = outcome
        assert result.stats.source_errors == {}
        assert result.ok is True


class TestDedupe:
    def test_duplicates_are_removed(self, outcome) -> None:
        _, result = outcome
        assert result.stats.duplicates_removed > 0

    def test_the_run_accounts_for_every_injected_duplicate(self, outcome) -> None:
        # 6 exact + 10 same-title + 12 rewritten per source, x2 sources = 56. The
        # collectors' identity guard takes the exact copies (same key as tier 1, and
        # it runs first) and the cascade takes the rest; the two must add to 56.
        _, result = outcome
        by_stage = result.stats.duplicates_by_stage
        assert by_stage.get("collector_identity", 0) == 12
        assert by_stage.get("title", 0) >= 20
        assert by_stage.get("near_duplicate", 0) >= 24
        assert result.stats.duplicates_removed == sum(by_stage.values())
        assert result.stats.duplicates_removed >= 56

    def test_the_cascade_still_runs_multiple_tiers(self) -> None:
        # A cascade where only the expensive tier ever runs is a cascade with dead
        # code, and the cheap tiers are what keep it fast.
        from pipeline.collectors import build_collectors
        from pipeline.dedup import Deduper

        settings = Settings(offline=True, cache_enabled=False)
        raw = []
        for collector in build_collectors(settings):
            raw.extend(collector.safe_collect(days=30).documents)
        result = Deduper().run(raw)
        assert {"title", "near_duplicate"} <= set(result.by_tier)

    def test_lsh_keeps_tier_three_off_the_quadratic_path(self) -> None:
        # 749 documents pairwise is ~280k comparisons; banding should keep it to a
        # few thousand, which is what makes the cascade viable as the corpus grows.
        from pipeline.collectors import build_collectors
        from pipeline.dedup import Deduper

        settings = Settings(offline=True, cache_enabled=False)
        raw = []
        for collector in build_collectors(settings):
            raw.extend(collector.safe_collect(days=30).documents)
        result = Deduper().run(raw)
        assert result.candidates_compared < len(raw) * 20

    def test_corpus_is_not_over_merged(self, outcome) -> None:
        # The threshold has to leave genuinely distinct documents alone; losing half
        # the corpus to a too-low threshold would still "pass" the floor test.
        _, result = outcome
        assert result.stats.duplicates_removed < result.stats.documents * 0.2


class TestAnchorsSurvive:
    def test_every_anchor_document_is_indexed(self, outcome) -> None:
        _, result = outcome
        urls = {document.url for document in result.documents}
        missing = [anchor.key for anchor in ANCHORS if anchor.url not in urls]
        assert missing == [], f"anchors lost during ingest: {missing}"

    def test_anchor_key_facts_survive_into_the_text(self, outcome) -> None:
        _, result = outcome
        by_url = {document.url: document for document in result.documents}
        quota = by_url["https://www.creg.com.cn/notice/2026-h1-rare-earth-quota"]
        assert "135000" in quota.text
        nickel = by_url["https://www.mining.com/indonesia-trims-nickel-ore-quota-for-2027/"]
        assert "240 million" in nickel.text


class TestIndexIntegrity:
    def test_index_round_trips_from_disk(self, outcome) -> None:
        settings, result = outcome
        store, embedder = load_index(settings)
        stats = store.stats()
        assert stats.documents == result.stats.documents
        assert stats.chunks == result.stats.chunks
        assert embedder.name == "hashing"

    def test_chunk_offsets_still_slice_their_source(self, outcome) -> None:
        # Every chunk must stay citable in place against the document it came from.
        # The window is stripped at its edges, so the invariant is equality modulo
        # surrounding whitespace rather than an exact length match.
        settings, result = outcome
        store, _ = load_index(settings)
        text_by_id = {document.id: document.text for document in result.documents}
        checked = 0
        for chunk in store.chunks():
            source = text_by_id.get(chunk.document_id)
            assert source is not None, f"chunk {chunk.id} has no source document"
            assert source[chunk.char_start : chunk.char_end].strip() == chunk.text.strip()
            checked += 1
        assert checked == len(store.chunks())

    def test_index_commodities_cover_the_priced_benchmarks(self, outcome) -> None:
        settings, _ = outcome
        store, _ = load_index(settings)
        assert {"lithium", "copper", "nickel", "zinc", "iron_ore"} <= store.available_commodities()

    def test_a_cross_lingual_question_finds_its_anchor(self, outcome) -> None:
        # The single most important end-to-end behaviour, and the one the pre-trim
        # used to break: a Chinese question against the English wire story.
        from pipeline.embedding import build_embedder
        from pipeline.retrieval import Retriever

        settings, _ = outcome
        store, _ = load_index(settings)
        retriever = Retriever(store, build_embedder())
        hits = retriever.retrieve("印度尼西亚 2027 年镍矿配额调整了多少？", top_k=5)
        urls = [hit.chunk.url for hit in hits]
        assert any("indonesia-trims-nickel-ore-quota" in url for url in urls)


class TestCli:
    def test_json_report_shape(self, outcome, capsys) -> None:
        settings, _ = outcome
        from pipeline.ingest import main

        exit_code = main(
            ["--days", "30", "--json", "--quiet", "--index-dir", str(settings.index_dir)]
        )
        assert exit_code == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["documents"] >= FLOOR_TOTAL
        assert payload["offline"] is True
        assert payload["per_kind"].keys() >= {"news", "policy", "price"}

    def test_missing_index_raises_a_helpful_error(self, tmp_path) -> None:
        from pipeline.ingest import IndexNotBuiltError, load_index

        settings = Settings(index_dir=tmp_path / "absent", cache_enabled=False)
        with pytest.raises(IndexNotBuiltError, match=r"pipeline\.ingest"):
            load_index(settings)
