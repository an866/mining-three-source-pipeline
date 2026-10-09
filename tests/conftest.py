"""Shared fixtures.

Every test runs with ``MINING_INDEX_DIR`` and ``MINING_CACHE_DIR`` pointed at
``tmp_path`` and with the settings cache reset, so nothing reads or writes the
repository's real ``.index``/``.cache`` and a developer's ``.env`` cannot change a
test outcome.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from pipeline.config import Settings, reset_settings_cache
from pipeline.embedding import HashingEmbedder
from pipeline.models import Chunk, Document, SourceKind
from pipeline.vectorstore import NumpyVectorStore

#: Small enough that embeddings are instant, large enough that cosine is not
#: degenerate.
TEST_DIM = 64


@pytest.fixture(autouse=True)
def _hermetic_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MINING_INDEX_DIR", str(tmp_path / "index"))
    monkeypatch.setenv("MINING_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("MINING_CACHE_ENABLED", "false")
    # An LLM key in the developer's environment must not switch the answerer.
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_EMBEDDING_API_KEY", raising=False)
    reset_settings_cache()
    yield
    reset_settings_cache()


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(index_dir=tmp_path / "index", cache_dir=tmp_path / "cache")


def make_document(
    *,
    doc_id: str,
    text: str,
    title: str | None = None,
    kind: SourceKind = SourceKind.NEWS,
    url: str = "",
    jurisdiction: str = "",
    commodity: str = "",
    source: str = "test",
    age_days: int = 1,
    meta: dict[str, Any] | None = None,
) -> Document:
    """Build a valid :class:`Document` with sensible defaults for tests."""
    return Document(
        id=doc_id,
        kind=kind,
        source=source,
        title=title or doc_id,
        url=url or f"https://example.test/{doc_id}",
        text=text,
        published_at=datetime.now(UTC) - timedelta(days=age_days),
        language="zh" if any("\u4e00" <= ch <= "\u9fff" for ch in text) else "en",
        jurisdiction=jurisdiction,
        commodity=commodity,
        content_hash=f"hash-{doc_id}",
        meta=meta or {},
    )


def index_documents(
    documents: list[Document], settings: Settings, *, dim: int = TEST_DIM
) -> tuple[NumpyVectorStore, HashingEmbedder]:
    """Chunk, embed and index ``documents`` into an in-memory store."""
    from pipeline.chunking import chunk_documents

    chunks: list[Chunk] = chunk_documents(documents, size=900, overlap=150)
    embedder = HashingEmbedder(dim=dim)
    store = NumpyVectorStore(embedder_name="hashing", dim=dim, settings=settings)
    store.add(chunks, embedder.embed([chunk.text for chunk in chunks]))
    return store, embedder


@pytest.fixture
def tiny_documents() -> list[Document]:
    """Three documents in two languages, with a planted rare term."""
    return [
        make_document(
            doc_id="nickel-id",
            title="Indonesia trims nickel ore quota for 2027",
            text=(
                "Indonesia's energy ministry cut the 2027 nickel ore production quota "
                "to 240 million wet metric tonnes from 275 million, a cut of roughly 13%. "
                "The ministry cited reserve depletion and environmental compliance."
            ),
            commodity="nickel",
            url="https://example.test/indonesia-nickel",
        ),
        make_document(
            doc_id="nickel-cn",
            title="关于调整镍管理政策的公告",
            text=(
                "商务部发布关于调整镍管理政策的公告，自 2026 年 9 月 1 日起施行。"
                "文件明确相关企业应当在 10 个工作日内完成申报。"
            ),
            kind=SourceKind.POLICY,
            jurisdiction="CN",
            commodity="nickel",
            url="https://example.test/cn-nickel",
        ),
        make_document(
            doc_id="copper-news",
            title="LME copper inventories fall to a six-week low",
            text=(
                "Exchange-registered copper inventories fell to their lowest level in six "
                "weeks, led by drawdowns at Asian warehouses on grid investment in China."
            ),
            commodity="copper",
            url="https://example.test/lme-copper",
        ),
    ]


@pytest.fixture
def tiny_store(
    tiny_documents: list[Document], settings: Settings
) -> tuple[NumpyVectorStore, HashingEmbedder]:
    return index_documents(tiny_documents, settings)


@pytest.fixture
def saved_index(tiny_documents: list[Document], settings: Settings) -> Settings:
    """A tiny index persisted at ``settings.index_dir``.

    The serving tests need an index *on disk* because the app loads it in its
    lifespan, exactly as it would in production -- injecting a store object would
    skip the code path most likely to break (id derivation, manifest, embedder
    mismatch).

    Built at ``settings.embedding_dim``, not the small test dimension: the app
    constructs its embedder from settings, so a 64-dim fixture index would trip the
    dimension guard on load (which is itself covered by a dedicated test).
    """
    store, _ = index_documents(tiny_documents, settings, dim=settings.embedding_dim)
    store.save(settings.index_dir)
    return settings
