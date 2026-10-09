"""Optional Chroma backend.

Selected with ``MINING_VECTOR_BACKEND=chroma``. Imported lazily by
:func:`pipeline.vectorstore.build_store` so that the default installation carries
no Chroma dependency.

It exists to answer a fair question: *why not just use Chroma?* The honest answer
is that for a corpus of this size Chroma adds a dependency, a persistence format
and an approximation without solving a problem we have -- see the scaling note in
``docs/architecture.md``. But the retrieval interface is the seam where a real
backend belongs once the corpus outgrows brute force, and having a second
implementation of that interface is what proves the seam is real rather than
aspirational: if only one class implements a Protocol, the Protocol is decoration.

Only the parts of Chroma that map onto this project's model are wrapped. Filters
are translated to a ``where`` clause, and the lexical half of the hybrid score is
recomputed locally because Chroma does not store the raw lexical statistics.
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from pipeline.config import Settings, get_settings
from pipeline.errors import ConfigurationError, VectorStoreError
from pipeline.logging_setup import get_logger
from pipeline.models import Chunk, IndexStats, QueryFilters
from pipeline.textutils import lexical_score
from pipeline.vectorstore import SearchHit, resolve_lexical_weight

__all__ = ["ChromaVectorStore"]

_log = get_logger(__name__)

_COLLECTION = "mining_chunks"


class ChromaVectorStore:
    """A thin adapter over a persistent Chroma collection."""

    backend = "chroma"

    def __init__(
        self, *, embedder_name: str = "", dim: int = 0, settings: Settings | None = None
    ) -> None:
        self._settings = settings or get_settings()
        self.embedder_name = embedder_name
        self.dim = dim
        try:
            import chromadb
        except ImportError as exc:  # pragma: no cover - depends on the install
            raise ConfigurationError(
                "MINING_VECTOR_BACKEND=chroma requires the 'chroma' extra: "
                "pip install 'mining-three-source-pipeline[chroma]'"
            ) from exc

        directory = Path(self._settings.index_dir) / "chroma"
        directory.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(directory))
        self._collection = self._client.get_or_create_collection(
            name=_COLLECTION,
            metadata={"hnsw:space": "cosine", "embedder": embedder_name or "unknown"},
        )

    # -- writing ------------------------------------------------------------

    def add(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        if len(chunks) != len(vectors):
            raise VectorStoreError(
                f"chunk/vector count mismatch: {len(chunks)} chunks, {len(vectors)} vectors"
            )
        if not chunks:
            return
        self._collection.upsert(
            ids=[chunk.id for chunk in chunks],
            embeddings=[list(map(float, vector)) for vector in vectors],
            documents=[chunk.text for chunk in chunks],
            metadatas=[_metadata(chunk) for chunk in chunks],
        )

    # -- reading ------------------------------------------------------------

    def search(
        self,
        query_vector: Sequence[float],
        *,
        query_text: str,
        top_k: int = 5,
        filters: QueryFilters | None = None,
        lexical_weight: float | None = None,
        candidate_k: int | None = None,
    ) -> list[SearchHit]:
        weight = resolve_lexical_weight(
            settings=self._settings,
            embedder_name=self.embedder_name,
            override=lexical_weight,
        )
        limit = candidate_k or self._settings.candidate_k
        where = _where_clause(filters)
        result = self._collection.query(
            query_embeddings=[list(map(float, query_vector))],
            n_results=max(limit, top_k),
            where=where or None,
            include=["documents", "metadatas", "distances"],
        )
        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        hits: list[SearchHit] = []
        for position, identifier in enumerate(ids):
            # Chroma's cosine *distance* is 1 - similarity.
            dense = max(0.0, 1.0 - float(distances[position]))
            text = str(documents[position] or "")
            lexical = lexical_score(query_text, text) if query_text else 0.0
            title = str((metadatas[position] or {}).get("title", ""))
            title_match = lexical_score(query_text, title) if query_text else 0.0
            hits.append(
                SearchHit(
                    chunk=_chunk_from(identifier, text, metadatas[position] or {}),
                    score=(
                        (1.0 - weight) * dense
                        + weight * lexical
                        + self._settings.title_weight * title_match
                    ),
                    vector_score=dense,
                    lexical_score=lexical,
                )
            )
        hits.sort(key=lambda hit: (-hit.score, hit.chunk.id))
        return hits[:top_k]

    # -- persistence --------------------------------------------------------

    def save(self, directory: Path | None = None) -> Path:
        # Chroma persists on every write; ``save`` exists to satisfy the protocol
        # and to report where the data lives.
        if directory is not None:
            return Path(directory)
        return Path(self._settings.index_dir) / "chroma"

    def load(self, directory: Path | None = None) -> bool:
        # ``int(...)`` because chromadb is untyped: without the cast the comparison
        # is ``Any > 0``, which mypy (rightly) refuses to call a bool.
        return int(self._collection.count()) > 0

    def available_commodities(self) -> set[str]:
        payload = self._collection.get(include=["metadatas"])
        return {
            str(metadata.get("commodity"))
            for metadata in (payload.get("metadatas") or [])
            if metadata.get("commodity")
        }

    def stats(self) -> IndexStats:
        payload = self._collection.get(include=["metadatas"])
        metadatas = payload.get("metadatas") or []
        # per_kind counts documents, not chunks (see IndexStats), so collapse by
        # document_id before tallying.
        kind_of: dict[str, str] = {}
        dates: list[datetime] = []
        for metadata in metadatas:
            document_id = str(metadata.get("document_id", ""))
            if document_id:
                kind_of[document_id] = str(metadata.get("kind", ""))
            stamp = metadata.get("published_at")
            if isinstance(stamp, str) and stamp:
                # A malformed timestamp must not lose the whole collection's stats.
                with suppress(ValueError):
                    dates.append(datetime.fromisoformat(stamp))
        per_kind: dict[str, int] = {}
        for kind in kind_of.values():
            per_kind[kind] = per_kind.get(kind, 0) + 1
        return IndexStats(
            documents=len(kind_of),
            chunks=int(self._collection.count()),
            dimensions=self.dim,
            embedder=self.embedder_name,
            backend=self.backend,
            per_kind=per_kind,
            earliest=min(dates) if dates else None,
            latest=max(dates) if dates else None,
        )


def _metadata(chunk: Chunk) -> dict[str, Any]:
    """Chroma metadata values must be scalars; ``None`` is not accepted."""
    return {
        "document_id": chunk.document_id,
        "ordinal": chunk.ordinal,
        "kind": chunk.kind.value,
        "source": chunk.source,
        "url": chunk.url,
        "title": chunk.title,
        "jurisdiction": chunk.jurisdiction,
        "commodity": chunk.commodity,
        "published_at": chunk.published_at.isoformat() if chunk.published_at else "",
        "char_start": chunk.char_start,
        "char_end": chunk.char_end,
    }


def _chunk_from(identifier: str, text: str, metadata: dict[str, Any]) -> Chunk:
    stamp = metadata.get("published_at")
    published: datetime | None = None
    if isinstance(stamp, str) and stamp:
        try:
            published = datetime.fromisoformat(stamp)
        except ValueError:
            published = None
    return Chunk(
        id=identifier,
        document_id=str(metadata.get("document_id", identifier.split(":", maxsplit=1)[0])),
        ordinal=int(metadata.get("ordinal", 0)),
        text=text,
        char_start=int(metadata.get("char_start", 0)),
        char_end=int(metadata.get("char_end", 0)),
        kind=str(metadata.get("kind", "news")),
        source=str(metadata.get("source", "")),
        url=str(metadata.get("url", "")),
        title=str(metadata.get("title", "")),
        published_at=published,
        jurisdiction=str(metadata.get("jurisdiction", "")),
        commodity=str(metadata.get("commodity", "")),
    )


def _where_clause(filters: QueryFilters | None) -> dict[str, Any]:
    """Translate :class:`QueryFilters` into a Chroma ``where`` clause.

    Returns ``{}`` (rather than ``None``) when nothing is set, so the caller can
    pass it through ``or None`` and Chroma treats it as "no filter".
    """
    if filters is None or filters.is_empty():
        return {}
    clauses: list[dict[str, Any]] = []
    if filters.kinds:
        kinds = [kind.value for kind in filters.kinds]
        if len(kinds) == 1:
            clauses.append({"kind": kinds[0]})
        else:
            clauses.append({"kind": {"$in": kinds}})
    if filters.jurisdiction:
        # Mirror ``NumpyVectorStore._mask``: jurisdiction scopes government
        # instruments, so a chunk that carries none (news, price) is not excluded.
        # Chroma has no "empty or equal" predicate, so it is spelled as an $or.
        clauses.append(
            {
                "$or": [
                    {"jurisdiction": filters.jurisdiction.upper()},
                    {"jurisdiction": ""},
                ]
            }
        )
    if filters.commodity:
        clauses.append({"commodity": filters.commodity})
    if filters.days:
        cutoff = (datetime.now(UTC) - timedelta(days=filters.days)).isoformat()
        clauses.append({"published_at": {"$gte": cutoff}})
    if not clauses:
        return {}
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}
