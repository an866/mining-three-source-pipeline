"""Vector store with hybrid (dense + lexical) retrieval.

Default backend is :class:`NumpyVectorStore`: an in-memory ``float32`` matrix, exact
cosine similarity, no service to run. At the corpus size this brief describes
(600-900 documents, a few thousand chunks) a brute-force exact search takes single
digit milliseconds, and an ANN index would be strictly worse -- it would add an
approximation, a dependency and a recall caveat in exchange for speed nobody
needs. The scaling note is in ``docs/architecture.md``; the short version is that
this class is the seam where HNSW or Chroma would go.

**Hybrid, not pure vector.** Dense retrieval with a feature-hashing embedder (or
any embedder) is weak on exactly the tokens a mining question turns on: ``62% Fe``,
``NI 43-101``, ``135,000 吨``, ``A$1.2bn``, ``碳酸锂``. Lexical overlap is strong
there and weak where dense retrieval is strong. The two are combined as
``(1 - w) * cosine + w * lexical``, and ``w`` is chosen from the embedder (see
:data:`_LEXICAL_WEIGHT_BY_EMBEDDER`) because the right balance is a property of the
embedder, not of the task: a learned model leads with dense similarity, whereas the
hashing embedder's "cosine" is itself surface overlap, so its precise lexical half
must lead. ``MINING_LEXICAL_WEIGHT`` pins an exact value when one is wanted.

**Filter before scoring.** ``kind``/``jurisdiction``/``commodity``/recency filters
are applied as a boolean mask on the matrix, so a question explicitly scoped to
"近 7 天的政策" cannot have its top-5 padded with price rows that merely scored
highly. Filtering after ranking is the classic way to end up with four results when
five were requested.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from pipeline.config import Settings, get_settings
from pipeline.errors import VectorStoreError
from pipeline.logging_setup import get_logger
from pipeline.models import Chunk, IndexStats, QueryFilters
from pipeline.textutils import lexical_score, search_units

__all__ = [
    "INDEX_VERSION",
    "NumpyVectorStore",
    "SearchHit",
    "VectorStore",
    "build_store",
    "resolve_lexical_weight",
]

_log = get_logger(__name__)

#: Bumped whenever the on-disk layout changes, so an old index is detected and
#: rebuilt instead of being read as garbage.
INDEX_VERSION = 1

_MANIFEST = "manifest.json"
_CHUNKS = "chunks.jsonl"
_VECTORS = "vectors.npy"

#: Lexical weight used when neither the caller nor the settings pin one, keyed by
#: embedder. The blend should reflect how much *orthogonal* signal the dense half
#: carries, and that depends entirely on the embedder:
#:
#: * ``openai`` (any learned model) encodes meaning, so dense similarity is the
#:   workhorse and lexical only breaks ties on the exact tokens a model blurs.
#: * ``hashing`` is a deterministic lexical retriever wearing a dense hat -- its
#:   "cosine" is feature-hashed surface overlap. On that path the two halves measure
#:   the same thing at two granularities, and the finer one (IDF-weighted, exact
#:   terms, cross-lingual concept markers) must lead. Measured on the 16-question
#:   probe, moving from 0.35 to 0.6 took top-5 recall from 15/16 to 16/16.
#:
#: Unknown embedders get the semantic default, which is the safer assumption: a
#: third-party embedder is more likely to be learned than hashed.
_LEXICAL_WEIGHT_BY_EMBEDDER: dict[str, float] = {"hashing": 0.6, "openai": 0.35}
_DEFAULT_LEXICAL_WEIGHT = 0.35

#: Candidate count below which the pure-cosine pre-trim is skipped entirely.
#: Lexical scoring is linear in the number of candidates, so at the corpus size the
#: brief targets this is microseconds -- and trimming first is a recall hazard (see
#: ``NumpyVectorStore.search``). The trim only earns its keep above this floor.
_PRE_TRIM_FLOOR = 2000


@dataclass(slots=True)
class SearchHit:
    """One ranked result, with the score decomposition kept for inspection."""

    chunk: Chunk
    score: float
    vector_score: float
    lexical_score: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk.id,
            "document_id": self.chunk.document_id,
            "score": round(self.score, 6),
            "vector_score": round(self.vector_score, 6),
            "lexical_score": round(self.lexical_score, 6),
        }


class VectorStore(Protocol):
    """The contract both backends satisfy."""

    backend: str

    def add(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        """Index ``chunks`` with their pre-computed ``vectors``."""

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
        """Return the best ``top_k`` chunks for one query."""

    def save(self, directory: Path | None = None) -> Path:
        """Persist the index; returns the directory written."""

    def load(self, directory: Path | None = None) -> bool:
        """Load the index; returns ``False`` when no usable index exists."""

    def stats(self) -> IndexStats:
        """What the store currently holds."""

    def available_commodities(self) -> set[str]:
        """Commodities with at least one indexed chunk.

        Read by the answer path's coverage check, which refuses a question about a
        mineral the index does not carry instead of answering it from the nearest
        evidence.
        """


class NumpyVectorStore:
    """Exact cosine + lexical hybrid over an in-memory matrix."""

    backend = "numpy"

    def __init__(
        self, *, embedder_name: str = "", dim: int = 0, settings: Settings | None = None
    ) -> None:
        self._settings = settings or get_settings()
        self.embedder_name = embedder_name
        self.dim = dim
        self._chunks: list[Chunk] = []
        self._matrix: np.ndarray | None = None
        # Document frequency per *unit*, for the IDF-weighted lexical half. It is
        # derived state, rebuilt from the chunks on load rather than persisted: at
        # this size the rebuild is one pass over the corpus, and a persisted copy is
        # one more thing that can disagree with the chunks it describes.
        #
        # The key vocabulary must be ``search_units`` -- the same one
        # ``lexical_score`` looks units up in. Indexing n-grams here while scoring on
        # maximal CJK runs would make every lookup miss, and a missing unit falls
        # back to the maximum IDF weight, which quietly turns the rare-term signal
        # this table exists for back into a constant.
        self._df: dict[str, int] = {}
        # Pre-computed unit sets, parallel to ``_chunks``. Scoring each candidate
        # would otherwise re-run the bilingual concept matchers on every query.
        self._text_units: list[set[str]] = []
        self._title_units: list[set[str]] = []

    # -- lexical statistics --------------------------------------------------

    def _index_tokens(self, chunk: Chunk) -> None:
        """Add one chunk's distinct units to the frequency table; cache its sets."""
        text_units = set(search_units(chunk.text))
        title_units = set(search_units(chunk.title))
        for unit in text_units | title_units:
            self._df[unit] = self._df.get(unit, 0) + 1
        self._text_units.append(text_units)
        self._title_units.append(title_units)

    def _rebuild_token_stats(self) -> None:
        self._df = {}
        self._text_units = []
        self._title_units = []
        for chunk in self._chunks:
            self._index_tokens(chunk)

    def _resolve_lexical_weight(self, override: float | None) -> float:
        return resolve_lexical_weight(
            settings=self._settings, embedder_name=self.embedder_name, override=override
        )

    @property
    def document_count(self) -> int:
        return len(self._chunks)

    # -- writing ------------------------------------------------------------

    def add(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        if len(chunks) != len(vectors):
            raise VectorStoreError(
                f"chunk/vector count mismatch: {len(chunks)} chunks, {len(vectors)} vectors"
            )
        if not chunks:
            return
        block = np.asarray(vectors, dtype=np.float32)
        if block.ndim != 2:
            raise VectorStoreError(f"expected a 2-D vector block, got shape {block.shape}")
        if self.dim and block.shape[1] != self.dim:
            raise VectorStoreError(
                f"embedding dimension changed: store holds {self.dim}, got {block.shape[1]}"
            )
        block = _l2_normalise(block)
        if self._matrix is None:
            self.dim = int(block.shape[1])
            self._matrix = block
        else:
            self._matrix = np.vstack([self._matrix, block])
        for chunk in chunks:
            self._index_tokens(chunk)
        self._chunks.extend(chunks)

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
        if self._matrix is None or not self._chunks:
            return []
        weight = self._resolve_lexical_weight(lexical_weight)
        limit = candidate_k or self._settings.candidate_k

        vector = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        if vector.shape[0] != self.dim:
            raise VectorStoreError(
                f"query vector has {vector.shape[0]} dimensions, index has {self.dim}"
            )
        query_unit = vector / (float(np.linalg.norm(vector)) or 1.0)

        # Mask first, score second: see the module docstring.
        indices = np.arange(len(self._chunks))
        mask = self._mask(filters)
        if mask is not None:
            indices = indices[mask]
        if indices.size == 0:
            return []
        if indices.size > max(limit, _PRE_TRIM_FLOOR):
            # Pre-trim with pure cosine so the lexical pass only runs on plausible
            # candidates. Lexical scoring is Python-level and, unlike the matrix
            # product, does not vectorise, so it is the expensive half.
            #
            # It only engages past ``_PRE_TRIM_FLOOR`` candidates, and that floor is
            # the point. Trimming is *pure cosine*, and pure cosine is exactly the
            # signal that fails on the queries the lexical half exists to rescue: a
            # Chinese question against an English document has almost no surface
            # overlap, so it ranks in the hundreds by cosine and a tight trim removes
            # it before its concept and exact-term matches are ever scored. Measured
            # on this corpus, the cross-lingual anchors sit at cosine rank 530-730 of
            # 764 -- a ``candidate_k`` of 50 deleted them outright. Scoring all 764
            # chunks lexically costs a few milliseconds, so the trim is only worth
            # its recall cost on corpora large enough to matter.
            coarse = self._matrix[indices] @ query_unit
            keep = np.argpartition(-coarse, min(limit, coarse.size - 1))[:limit]
            indices = indices[np.sort(keep)]

        similarities = self._matrix[indices] @ query_unit
        title_weight = self._settings.title_weight
        n_docs = len(self._chunks)
        hits: list[SearchHit] = []
        for position, chunk_index in enumerate(indices):
            index = int(chunk_index)
            chunk = self._chunks[index]
            dense = float(max(0.0, similarities[position]))
            lexical = (
                lexical_score(
                    query_text,
                    chunk.text,
                    df=self._df,
                    n_docs=n_docs,
                    doc_units=self._text_units[index],
                )
                if query_text
                else 0.0
            )
            # Field boost. Mentioning many concepts lowers a document's cosine
            # similarity under L2 normalisation, which is exactly the wrong
            # incentive: the short document whose *title* is the question should win
            # over the long one that merely lists every mineral in passing. IDF is
            # deliberately *not* applied to the title: a title is ten tokens long,
            # and length normalisation against corpus statistics adds noise rather
            # than signal at that scale.
            title_match = (
                lexical_score(query_text, chunk.title, doc_units=self._title_units[index])
                if query_text
                else 0.0
            )
            hits.append(
                SearchHit(
                    chunk=chunk,
                    score=(1.0 - weight) * dense + weight * lexical + title_weight * title_match,
                    vector_score=dense,
                    lexical_score=lexical,
                )
            )
        # Sort by score, then by chunk id so equal scores rank deterministically --
        # an unstable order would make recall@5 wobble between identical runs.
        hits.sort(key=lambda hit: (-hit.score, hit.chunk.id))
        return hits[:top_k]

    def chunks(self) -> list[Chunk]:
        """Every indexed chunk, in insertion order (used by the eval harness)."""
        return list(self._chunks)

    def available_commodities(self) -> set[str]:
        return {chunk.commodity for chunk in self._chunks if chunk.commodity}

    def _mask(self, filters: QueryFilters | None) -> np.ndarray | None:
        """A boolean mask over the chunk list, or ``None`` when nothing is filtered."""
        if filters is None or filters.is_empty():
            return None
        keep = np.ones(len(self._chunks), dtype=bool)
        kinds = {kind.value for kind in filters.kinds}
        cutoff: datetime | None = None
        if filters.days:
            cutoff = datetime.now(UTC) - timedelta(days=filters.days)

        for index, chunk in enumerate(self._chunks):
            if kinds and chunk.kind.value not in kinds:
                keep[index] = False
                continue
            # Jurisdiction scopes *government instruments*, not wire copy. Only policy
            # chunks carry one; news and price chunks have an empty jurisdiction, and
            # dropping them would answer "中国铜冶炼厂加工费" -- where 中国 is the
            # subject -- by deleting the English news story that reports the result.
            # A chunk with no jurisdiction is therefore never excluded by this filter.
            if (
                filters.jurisdiction
                and chunk.jurisdiction
                and chunk.jurisdiction.upper() != filters.jurisdiction.upper()
            ):
                keep[index] = False
                continue
            if filters.commodity and chunk.commodity != filters.commodity:
                keep[index] = False
                continue
            if filters.source and filters.source.lower() not in chunk.source.lower():
                keep[index] = False
                continue
            if cutoff is not None:
                published = chunk.published_at
                if published is not None:
                    if published.tzinfo is None:
                        published = published.replace(tzinfo=UTC)
                    if published < cutoff:
                        keep[index] = False
        return keep

    # -- persistence --------------------------------------------------------

    def save(self, directory: Path | None = None) -> Path:
        target = Path(directory) if directory is not None else self._settings.index_dir
        target.mkdir(parents=True, exist_ok=True)
        if self._matrix is None:
            # Write a coherent empty index rather than leaving a half-written
            # directory behind; ``load`` must never see a matrix without chunks.
            np.save(target / _VECTORS, np.zeros((0, self.dim or 1), dtype=np.float32))
        else:
            np.save(target / _VECTORS, self._matrix)
        with (target / _CHUNKS).open("w", encoding="utf-8") as handle:
            for chunk in self._chunks:
                handle.write(json.dumps(chunk.model_dump(mode="json"), ensure_ascii=False) + "\n")
        manifest = {
            "version": INDEX_VERSION,
            "backend": self.backend,
            "embedder": self.embedder_name,
            "dim": self.dim,
            "chunks": len(self._chunks),
            "written_at": datetime.now(UTC).isoformat(),
        }
        (target / _MANIFEST).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        _log.info(
            "index written",
            extra={"extra_fields": {"directory": str(target), "chunks": len(self._chunks)}},
        )
        return target

    def load(self, directory: Path | None = None) -> bool:
        target = Path(directory) if directory is not None else self._settings.index_dir
        manifest_path = target / _MANIFEST
        if not manifest_path.is_file():
            return False
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise VectorStoreError(f"unreadable manifest at {manifest_path}: {exc}") from exc

        if manifest.get("version") != INDEX_VERSION:
            raise VectorStoreError(
                f"index at {target} has version {manifest.get('version')}, "
                f"expected {INDEX_VERSION}; rebuild with `python -m pipeline.ingest`"
            )
        if self.embedder_name and manifest.get("embedder") not in ("", self.embedder_name):
            raise VectorStoreError(
                f"index was built with embedder {manifest.get('embedder')!r}, "
                f"configured embedder is {self.embedder_name!r}; rebuild the index"
            )

        chunks: list[Chunk] = []
        try:
            with (target / _CHUNKS).open("r", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        chunks.append(Chunk.model_validate(json.loads(line)))
            matrix = np.load(target / _VECTORS)
        except FileNotFoundError as exc:
            raise VectorStoreError(f"index at {target} is incomplete: {exc}") from exc
        except (json.JSONDecodeError, ValueError) as exc:
            raise VectorStoreError(f"index at {target} is corrupt: {exc}") from exc

        matrix = np.asarray(matrix, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != len(chunks):
            raise VectorStoreError(
                f"index at {target} is inconsistent: {matrix.shape[0]} vectors, "
                f"{len(chunks)} chunks"
            )
        # An explicitly-requested dimension must match the stored one. Without this
        # the mismatch only surfaces on the first query, as "query vector has 512
        # dimensions, index has 64" from inside the search -- which reads like a bug
        # in the caller rather than a stale index. It is a configuration error and
        # belongs at startup.
        if self.dim and matrix.size and int(matrix.shape[1]) != self.dim:
            raise VectorStoreError(
                f"index at {target} was built with {matrix.shape[1]}-dimensional "
                f"vectors but the configured embedder produces {self.dim}; rebuild "
                f"with `python -m pipeline.ingest`"
            )
        self._chunks = chunks
        self._matrix = matrix
        self._rebuild_token_stats()
        self.dim = int(matrix.shape[1]) if matrix.size else int(manifest.get("dim", 0))
        self.embedder_name = str(manifest.get("embedder", self.embedder_name))
        return True

    # -- reporting ----------------------------------------------------------

    def stats(self) -> IndexStats:
        # per_kind counts documents, not chunks (see IndexStats): a document's chunks
        # all carry its kind, so the last write per document_id is authoritative.
        kind_of: dict[str, str] = {}
        dates: list[datetime] = []
        for chunk in self._chunks:
            kind_of[chunk.document_id] = chunk.kind.value
            if chunk.published_at is not None:
                published = chunk.published_at
                dates.append(published if published.tzinfo else published.replace(tzinfo=UTC))
        per_kind: dict[str, int] = {}
        for kind in kind_of.values():
            per_kind[kind] = per_kind.get(kind, 0) + 1
        return IndexStats(
            documents=len(kind_of),
            chunks=len(self._chunks),
            dimensions=self.dim,
            embedder=self.embedder_name,
            backend=self.backend,
            per_kind=per_kind,
            earliest=min(dates) if dates else None,
            latest=max(dates) if dates else None,
        )


def resolve_lexical_weight(
    *,
    settings: Settings,
    embedder_name: str,
    override: float | None = None,
) -> float:
    """Blend weight for the lexical half: caller > settings > embedder default.

    Shared by both backends so that ``numpy`` and ``chroma`` cannot drift into
    scoring the same query differently.
    """
    chosen = override if override is not None else settings.lexical_weight
    if chosen is None:
        chosen = _LEXICAL_WEIGHT_BY_EMBEDDER.get(embedder_name, _DEFAULT_LEXICAL_WEIGHT)
    return min(1.0, max(0.0, chosen))


def _l2_normalise(matrix: np.ndarray) -> np.ndarray:
    """Row-normalise, leaving zero rows as zeros.

    Normalising at insert time means the search hot path is a single matrix
    multiply. Dividing there instead would recompute the same norms on every query.
    """
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    # ``asarray`` re-states the type: numpy's ``__truediv__`` stubs return ``Any``,
    # and mypy --strict will not accept ``Any`` from a function declared ``ndarray``.
    return np.asarray(matrix / norms)


def build_store(
    backend: str | None = None,
    *,
    settings: Settings | None = None,
    embedder_name: str = "",
    dim: int = 0,
) -> VectorStore:
    """Construct the configured backend.

    ``chroma`` is imported lazily so that the default install has no Chroma
    dependency; asking for it without it installed is a configuration error with a
    clear remedy, not an ``ImportError`` from three frames deep.
    """
    resolved = settings or get_settings()
    choice = (backend or resolved.vector_backend).strip().lower()
    if choice == "numpy":
        return NumpyVectorStore(embedder_name=embedder_name, dim=dim, settings=resolved)
    if choice == "chroma":
        from pipeline.vectorstore_chroma import ChromaVectorStore

        return ChromaVectorStore(embedder_name=embedder_name, dim=dim, settings=resolved)
    raise VectorStoreError(
        f"unknown MINING_VECTOR_BACKEND={choice!r}; expected 'numpy' or 'chroma'"
    )
