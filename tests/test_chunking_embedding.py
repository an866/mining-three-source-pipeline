"""Chunking (offsets must be real) and embedding (deterministic, concept-aware)."""

from __future__ import annotations

import math

from tests.conftest import make_document

from pipeline.chunking import chunk_document, chunk_documents, chunk_stats
from pipeline.embedding import CachedEmbedder, HashingEmbedder
from pipeline.models import SourceKind


class TestChunkOffsets:
    def test_offsets_slice_the_original_text(self) -> None:
        text = "第一句话。" * 60
        doc = make_document(doc_id="d", text=text)
        chunks = chunk_document(doc, size=120, overlap=30)
        assert chunks
        for chunk in chunks:
            # A citation that cannot be re-read in place is not a citation.
            assert text[chunk.char_start : chunk.char_end] == chunk.text

    def test_chunks_cover_the_whole_document(self) -> None:
        text = "Lithium destocking nears its end. " * 40
        doc = make_document(doc_id="d", text=text)
        chunks = chunk_document(doc, size=200, overlap=50)
        assert chunks[0].char_start == 0
        # Trailing whitespace is trimmed from the final window, so the cover is of
        # the stripped text -- and a citation still re-reads correctly.
        assert text[: chunks[-1].char_end].strip() == text.strip()

    def test_ordinals_are_sequential_and_ids_unique(self) -> None:
        doc = make_document(doc_id="d", text="copper " * 400)
        chunks = chunk_document(doc, size=100, overlap=20)
        assert [c.ordinal for c in chunks] == list(range(len(chunks)))
        assert len({c.id for c in chunks}) == len(chunks)

    def test_short_document_is_a_single_chunk(self) -> None:
        doc = make_document(doc_id="d", text="Zinc charges fell to multi-year lows.")
        chunks = chunk_document(doc, size=900, overlap=150)
        assert len(chunks) == 1
        assert chunks[0].text == doc.text

    def test_filter_fields_are_denormalised_onto_the_chunk(self) -> None:
        # The store filters on chunks, so a chunk that does not carry its document's
        # jurisdiction/commodity would be unfilterable.
        doc = make_document(
            doc_id="d",
            text="镍 " * 500,
            kind=SourceKind.POLICY,
            jurisdiction="CN",
            commodity="nickel",
            url="https://x.test/p",
        )
        chunks = chunk_document(doc, size=100, overlap=20)
        assert all(c.jurisdiction == "CN" for c in chunks)
        assert all(c.commodity == "nickel" for c in chunks)
        assert all(c.document_id == "d" for c in chunks)
        assert all(c.url == "https://x.test/p" for c in chunks)

    def test_chunk_documents_flattens_in_order(self) -> None:
        docs = [
            make_document(doc_id="a", text="copper " * 300),
            make_document(doc_id="b", text="zinc " * 300),
        ]
        chunks = chunk_documents(docs, size=200, overlap=0)
        ids = [c.document_id for c in chunks]
        assert ids == sorted(ids, key=lambda value: (value != "a",))

    def test_stats_are_reported(self) -> None:
        docs = [make_document(doc_id="a", text="copper " * 300)]
        summary = chunk_stats(chunk_documents(docs, size=200, overlap=0))
        assert summary["count"] >= 1
        assert summary["min_chars"] <= summary["mean_chars"] <= summary["max_chars"]


class TestHashingEmbedder:
    def test_dimension_and_unit_norm(self) -> None:
        embedder = HashingEmbedder(dim=128)
        vector = embedder.embed_query("copper inventories fall")
        assert len(vector) == 128
        assert math.isclose(math.sqrt(sum(v * v for v in vector)), 1.0, rel_tol=1e-6)

    def test_deterministic_across_instances(self) -> None:
        text = "Indonesia trims nickel ore quota for 2027"
        assert HashingEmbedder(dim=64).embed_query(text) == HashingEmbedder(dim=64).embed_query(
            text
        )

    def test_empty_text_does_not_produce_a_zero_vector(self) -> None:
        # A zero vector normalises to NaN and would compare equal to nothing.
        vector = HashingEmbedder(dim=32).embed_query("")
        assert any(value != 0.0 for value in vector)

    def test_concepts_lift_cross_lingual_similarity(self) -> None:
        embedder = HashingEmbedder(dim=256)
        chinese = embedder.embed_query("印度尼西亚镍矿配额")
        english = embedder.embed_query("Indonesia nickel ore quota")
        unrelated = embedder.embed_query("zinc treatment charges collapse")
        bridge = sum(a * b for a, b in zip(chinese, english, strict=True))
        other = sum(a * b for a, b in zip(chinese, unrelated, strict=True))
        assert bridge > other

    def test_same_text_is_identical_regardless_of_case(self) -> None:
        embedder = HashingEmbedder(dim=64)
        assert embedder.embed_query("Copper Rose") == embedder.embed_query("copper rose")

    def test_embed_handles_a_batch(self) -> None:
        vectors = HashingEmbedder(dim=32).embed(["a copper note", "a zinc note"])
        assert len(vectors) == 2
        assert vectors[0] != vectors[1]


class TestCachedEmbedder:
    def test_second_call_is_served_from_cache(self, tmp_path) -> None:
        from pipeline.cache import DiskCache
        from pipeline.config import Settings

        # The suite disables the cache globally; this test is the one that needs it.
        enabled = Settings(cache_dir=tmp_path / "cache", cache_enabled=True)
        calls = {"n": 0}

        class Counting(HashingEmbedder):
            def embed(self, texts):  # type: ignore[override]
                calls["n"] += len(texts)
                return super().embed(texts)

        embedder = CachedEmbedder(Counting(dim=32), DiskCache(settings=enabled))
        embedder.embed(["same text", "same text"])
        assert calls["n"] == 2
        embedder.embed(["same text", "same text"])
        assert calls["n"] == 2, "a warm cache must not recompute"

    def test_wrong_dimension_in_cache_is_ignored(self, tmp_path) -> None:
        from pipeline.cache import DiskCache
        from pipeline.config import Settings

        enabled = Settings(cache_dir=tmp_path / "cache", cache_enabled=True)
        cache = DiskCache(settings=enabled)
        embedder = CachedEmbedder(HashingEmbedder(dim=32), cache)
        cache.set("embeddings", embedder._key("hello"), [0.0] * 8)
        vector = embedder.embed(["hello"])[0]
        assert len(vector) == 32

    def test_name_and_dim_are_forwarded(self) -> None:
        embedder = CachedEmbedder(HashingEmbedder(dim=48))
        assert embedder.name == "hashing"
        assert embedder.dim == 48
