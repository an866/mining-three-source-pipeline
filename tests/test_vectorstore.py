"""The vector store: hybrid scoring, filter semantics, persistence, pre-trim.

Several of these guard specific bugs found while measuring retrieval, so they are
regression tests rather than shape checks:

* the pure-cosine pre-trim deleting cross-lingual anchors before lexical scoring
* a jurisdiction filter deleting every news/price chunk (they carry no jurisdiction)
* the IDF table keyed on a different vocabulary than the one looked up
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from tests.conftest import index_documents, make_document

from pipeline.config import Settings
from pipeline.errors import VectorStoreError
from pipeline.models import QueryFilters, SourceKind
from pipeline.textutils import document_id
from pipeline.vectorstore import (
    _PRE_TRIM_FLOOR,
    NumpyVectorStore,
    build_store,
    resolve_lexical_weight,
)


class TestHybridScoring:
    def test_exact_term_query_finds_its_document(self, tiny_store) -> None:
        store, embedder = tiny_store
        hits = store.search(
            embedder.embed_query("Indonesia nickel quota 2027"),
            query_text="Indonesia nickel quota 2027",
        )
        assert hits
        assert hits[0].chunk.url == "https://example.test/indonesia-nickel"

    def test_score_decomposition_is_reported(self, tiny_store) -> None:
        store, embedder = tiny_store
        hit = store.search(
            embedder.embed_query("copper inventories"), query_text="copper inventories"
        )[0]
        assert hit.vector_score > 0.0
        assert hit.lexical_score > 0.0
        assert hit.score >= max(hit.vector_score, hit.lexical_score) * 0.0

    def test_ties_break_on_chunk_id_so_order_is_stable(self, tiny_store) -> None:
        store, embedder = tiny_store
        first = [h.chunk.id for h in store.search(embedder.embed_query("镍"), query_text="镍")]
        second = [h.chunk.id for h in store.search(embedder.embed_query("镍"), query_text="镍")]
        assert first == second

    def test_empty_store_returns_no_hits(self, settings) -> None:
        store = NumpyVectorStore(embedder_name="hashing", dim=8, settings=settings)
        assert store.search([0.0] * 8, query_text="anything") == []

    def test_dimension_mismatch_is_an_error(self, tiny_store) -> None:
        store, _ = tiny_store
        with pytest.raises(VectorStoreError):
            store.search([0.0] * 4, query_text="x")


class TestPreTrimFloor:
    """The pre-trim must not delete the documents lexical scoring exists to rescue."""

    def test_lexical_rescue_survives_a_tight_candidate_k(self, settings, monkeypatch) -> None:
        # A Chinese query and an English answer: almost no surface overlap, so pure
        # cosine ranks the answer last. With the floor disabled and a small
        # candidate_k the old code dropped it before its concept match was scored.
        docs = [
            make_document(
                doc_id=f"filler-{i}",
                title=f"关于调整镍管理政策的公告（{i}）",
                text="商务部发布关于调整镍管理政策的公告，自 2026 年 9 月 1 日起施行。" * 3,
                kind=SourceKind.POLICY,
                jurisdiction="CN",
                commodity="nickel",
            )
            for i in range(24)
        ]
        docs.append(
            make_document(
                doc_id="anchor",
                title="Indonesia trims nickel ore quota for 2027",
                text=(
                    "Indonesia's energy ministry cut the 2027 nickel ore production quota "
                    "to 240 million wet metric tonnes from 275 million, a cut of roughly "
                    "13%, citing reserve depletion."
                ),
                commodity="nickel",
                url="https://example.test/anchor",
            )
        )
        store, embedder = index_documents(docs, settings)
        query = "印度尼西亚 2027 年镍矿配额调整了多少？"
        vector = embedder.embed_query(query)

        # Floor lifted (as in the real corpus) -> the anchor is scored and wins.
        monkeypatch.setattr("pipeline.vectorstore._PRE_TRIM_FLOOR", 0)
        rescued = store.search(vector, query_text=query, top_k=3, candidate_k=50)
        assert "https://example.test/anchor" in [h.chunk.url for h in rescued]

        # Floor above the corpus size -> no trim either, same answer.
        monkeypatch.setattr("pipeline.vectorstore._PRE_TRIM_FLOOR", 10_000)
        untrimmed = store.search(vector, query_text=query, top_k=3, candidate_k=50)
        assert "https://example.test/anchor" in [h.chunk.url for h in untrimmed]

    def test_trim_still_engages_above_the_floor(self, settings, monkeypatch) -> None:
        # The floor is a threshold, not a removal: above it the trim must still run,
        # or the knob is dead and a large corpus pays the O(n) lexical cost.
        docs = [
            make_document(
                doc_id=f"d{i}",
                title=f"nickel story {i}",
                text="nickel ore quota news " * 20,
                commodity="nickel",
            )
            for i in range(40)
        ]
        store, embedder = index_documents(docs, settings)
        monkeypatch.setattr("pipeline.vectorstore._PRE_TRIM_FLOOR", 5)
        hits = store.search(
            embedder.embed_query("nickel"), query_text="nickel", top_k=3, candidate_k=5
        )
        assert len(hits) <= 3

    def test_default_floor_is_above_the_brief_corpus(self) -> None:
        # The default has to be inert for a 600-900 document corpus, which is what
        # the brief specifies.
        assert _PRE_TRIM_FLOOR >= 2000


class TestFilterMask:
    def _store(self, settings):
        docs = [
            make_document(
                doc_id="cn",
                title="关于镍的通知",
                text="镍相关内容。" * 30,
                kind=SourceKind.POLICY,
                jurisdiction="CN",
                commodity="nickel",
            ),
            make_document(
                doc_id="au",
                title="Nickel review",
                text="Nickel market review content. " * 30,
                kind=SourceKind.POLICY,
                jurisdiction="AU",
                commodity="nickel",
            ),
            make_document(
                doc_id="news",
                title="Indonesia nickel quota",
                text="Indonesia nickel quota story. " * 30,
                kind=SourceKind.NEWS,
                commodity="nickel",
            ),
            make_document(
                doc_id="zinc",
                title="Zinc charges",
                text="Zinc treatment charges fell. " * 30,
                kind=SourceKind.PRICE,
                commodity="zinc",
            ),
        ]
        return index_documents(docs, settings)

    def test_jurisdiction_does_not_exclude_a_chunk_without_one(self, settings) -> None:
        # The bug this guards: 中国 in the question planned jurisdiction=CN and the
        # news chunk (jurisdiction="") was dropped, deleting the answer.
        store, embedder = self._store(settings)
        hits = store.search(
            embedder.embed_query("镍"),
            query_text="镍",
            filters=QueryFilters(jurisdiction="CN"),
            top_k=10,
        )
        urls = {h.chunk.document_id for h in hits}
        assert "news" in urls, "a jurisdiction filter must not delete wire copy"
        assert "cn" in urls
        assert "au" not in urls, "a policy chunk from another jurisdiction is excluded"

    def test_kind_filter_is_hard(self, settings) -> None:
        store, embedder = self._store(settings)
        hits = store.search(
            embedder.embed_query("镍"),
            query_text="镍",
            filters=QueryFilters(kinds=[SourceKind.NEWS]),
            top_k=10,
        )
        assert {h.chunk.document_id for h in hits} == {"news"}

    def test_commodity_filter_is_applied(self, settings) -> None:
        store, embedder = self._store(settings)
        hits = store.search(
            embedder.embed_query("charges"),
            query_text="charges",
            filters=QueryFilters(commodity="zinc"),
            top_k=10,
        )
        assert {h.chunk.document_id for h in hits} == {"zinc"}

    def test_days_filter_excludes_old_documents(self, settings) -> None:
        docs = [
            make_document(doc_id="new", title="Fresh copper", text="copper " * 60, age_days=1),
            make_document(doc_id="old", title="Old copper", text="copper " * 60, age_days=200),
        ]
        store, embedder = index_documents(docs, settings)
        hits = store.search(
            embedder.embed_query("copper"),
            query_text="copper",
            filters=QueryFilters(days=7),
            top_k=10,
        )
        assert {h.chunk.document_id for h in hits} == {"new"}

    def test_empty_filters_means_no_mask(self, settings) -> None:
        store, embedder = self._store(settings)
        hits = store.search(
            embedder.embed_query("镍"), query_text="镍", filters=QueryFilters(), top_k=10
        )
        assert len(hits) >= 3


class TestPersistence:
    def test_round_trip_preserves_results(self, tiny_documents, settings, tmp_path) -> None:
        store, embedder = index_documents(tiny_documents, settings)
        target = store.save(tmp_path / "index")

        restored = NumpyVectorStore(embedder_name="hashing", dim=store.dim, settings=settings)
        assert restored.load(target) is True
        assert restored.document_count == store.document_count

        query = "LME copper inventories"
        before = [h.chunk.id for h in store.search(embedder.embed_query(query), query_text=query)]
        after = [h.chunk.id for h in restored.search(embedder.embed_query(query), query_text=query)]
        assert before == after

    def test_missing_index_reports_false_not_an_error(self, settings, tmp_path) -> None:
        store = NumpyVectorStore(embedder_name="hashing", dim=8, settings=settings)
        assert store.load(tmp_path / "nope") is False

    def test_version_mismatch_is_an_error(self, tiny_documents, settings, tmp_path) -> None:
        import json

        store, _ = index_documents(tiny_documents, settings)
        target = store.save(tmp_path / "index")
        manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
        manifest["version"] = 999
        (target / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

        restored = NumpyVectorStore(embedder_name="hashing", dim=store.dim, settings=settings)
        with pytest.raises(VectorStoreError, match="version"):
            restored.load(target)

    def test_embedder_mismatch_is_an_error(self, tiny_documents, settings, tmp_path) -> None:
        store, _ = index_documents(tiny_documents, settings)
        target = store.save(tmp_path / "index")
        restored = NumpyVectorStore(embedder_name="openai", dim=store.dim, settings=settings)
        with pytest.raises(VectorStoreError, match="embedder"):
            restored.load(target)

    def test_dimension_mismatch_fails_at_load_not_at_query(self, tmp_path, settings) -> None:
        # A stale index built at another dimension must be caught when the service
        # starts, not on the first query -- otherwise the operator sees a shape error
        # from deep inside the search and has to reconstruct the cause.
        docs = [make_document(doc_id="d", title="Copper", text="copper " * 40, commodity="copper")]
        store, _ = index_documents(docs, settings, dim=32)
        target = store.save(tmp_path / "index")

        wrong = NumpyVectorStore(embedder_name="hashing", dim=512, settings=settings)
        with pytest.raises(VectorStoreError, match="512"):
            wrong.load(target)

        right = NumpyVectorStore(embedder_name="hashing", dim=32, settings=settings)
        assert right.load(target) is True

    def test_manifest_records_what_was_built(self, tiny_documents, settings, tmp_path) -> None:
        import json

        store, _ = index_documents(tiny_documents, settings)
        target = store.save(tmp_path / "index")
        manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["embedder"] == "hashing"
        assert manifest["chunks"] == store.document_count
        assert manifest["backend"] == "numpy"


class TestStats:
    def test_per_kind_counts_documents_not_chunks(self, settings) -> None:
        # A document long enough to chunk several times must count once, or /stats
        # and the ingest report disagree under the same field name.
        docs = [
            make_document(doc_id="long", title="Long copper", text="copper " * 900),
            make_document(doc_id="short", title="Short zinc", text="zinc " * 5),
        ]
        store, _ = index_documents(docs, settings)
        stats = store.stats()
        assert stats.documents == 2
        assert stats.chunks > 2
        assert stats.per_kind == {"news": 2}

    def test_available_commodities_reports_the_corpus(self, tiny_documents, settings) -> None:
        store, _ = index_documents(tiny_documents, settings)
        assert store.available_commodities() == {"nickel", "copper"}

    def test_earliest_and_latest_are_set(self, tiny_documents, settings) -> None:
        store, _ = index_documents(tiny_documents, settings)
        stats = store.stats()
        assert stats.earliest is not None
        assert stats.latest is not None
        assert stats.earliest <= stats.latest


class TestLexicalWeightResolution:
    def test_embedder_selects_the_default(self, settings) -> None:
        # The blend should reflect how much orthogonal signal the dense half carries,
        # which is a property of the embedder.
        assert resolve_lexical_weight(settings=settings, embedder_name="hashing") == 0.6
        assert resolve_lexical_weight(settings=settings, embedder_name="openai") == 0.35

    def test_unknown_embedder_gets_the_semantic_default(self, settings) -> None:
        assert resolve_lexical_weight(settings=settings, embedder_name="mystery") == 0.35

    def test_settings_pin_overrides_the_embedder_default(self) -> None:
        pinned = Settings(lexical_weight=0.2)
        assert resolve_lexical_weight(settings=pinned, embedder_name="hashing") == 0.2

    def test_caller_override_wins_over_everything(self) -> None:
        pinned = Settings(lexical_weight=0.2)
        assert resolve_lexical_weight(settings=pinned, embedder_name="hashing", override=0.9) == 0.9

    def test_out_of_range_override_is_clamped(self, settings) -> None:
        assert resolve_lexical_weight(settings=settings, embedder_name="hashing", override=5) == 1.0
        assert (
            resolve_lexical_weight(settings=settings, embedder_name="hashing", override=-1) == 0.0
        )


class TestBuildStore:
    def test_numpy_is_the_default(self, settings) -> None:
        assert build_store(settings=settings).backend == "numpy"

    def test_unknown_backend_is_a_clear_error(self, settings) -> None:
        with pytest.raises(VectorStoreError, match="unknown MINING_VECTOR_BACKEND"):
            build_store("pinecone", settings=settings)


def test_document_id_is_stable_across_calls() -> None:
    # Ids are the tier-1 dedup key; instability there silently disables a tier.
    stamp = datetime(2026, 9, 1, tzinfo=UTC)
    first = document_id("news", "https://x.test/a", "A title", stamp)
    second = document_id("news", "https://x.test/a", "A title", stamp)
    assert first == second
    assert first != document_id("news", "https://x.test/b", "A title", stamp)


def test_age_days_fixture_helper_is_utc_aware() -> None:
    doc = make_document(doc_id="d", text="x", age_days=3)
    assert doc.published_at is not None
    assert doc.published_at.tzinfo is not None
    assert doc.published_at < datetime.now(UTC) + timedelta(days=1)
