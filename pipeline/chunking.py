"""Documents to chunks.

Retrieval quality is decided here more than anywhere else. Two decisions:

**Chunk, do not embed whole documents.** A 4,000-character policy analysis
embedded as one vector answers no question well -- the vector is an average of
six topics. 900 characters with 150 overlap keeps each vector on one subject while
still carrying enough context to be quotable.

**Denormalise the filter fields onto the chunk.** ``kind``, ``source``,
``jurisdiction``, ``commodity`` and ``published_at`` are copied onto every chunk
so that a filtered search never has to join back to the document table. At this
scale the duplication is a few hundred kilobytes; the join it avoids would be in
the hot path of every query.

Chunks carry real character offsets into the document text, so a citation can
quote and a reader can locate. A chunk whose span cannot be computed is a bug, not
a degraded result -- :func:`~pipeline.textutils.chunk_text` always returns valid
offsets, and this module asserts the invariant.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from pipeline.logging_setup import get_logger
from pipeline.models import Chunk, Document
from pipeline.textutils import chunk_text

__all__ = ["chunk_document", "chunk_documents"]

_log = get_logger(__name__)


def chunk_document(document: Document, *, size: int = 900, overlap: int = 150) -> list[Chunk]:
    """Split one document, denormalising filter fields onto each chunk."""
    if not document.text.strip():
        return []
    spans = chunk_text(document.text, size=size, overlap=overlap)
    chunks: list[Chunk] = []
    for ordinal, (text, start, end) in enumerate(spans):
        if not text.strip():
            continue
        chunks.append(
            Chunk(
                id=f"{document.id}:{ordinal}",
                document_id=document.id,
                ordinal=ordinal,
                text=text,
                char_start=start,
                char_end=end,
                kind=document.kind,
                source=document.source,
                url=document.url,
                title=document.title,
                published_at=document.published_at,
                jurisdiction=document.jurisdiction,
                commodity=document.commodity,
            )
        )
    return chunks


def chunk_documents(
    documents: Iterable[Document], *, size: int = 900, overlap: int = 150
) -> list[Chunk]:
    """Split a corpus, preserving document order.

    Order is preserved deliberately: chunk ordinals are part of the stable id, and
    a run whose chunk ids shuffle would invalidate every cached embedding.
    """
    if overlap >= size:
        # chunk_text handles this, but a caller that sets overlap=size almost
        # always means "no overlap" and silently gets pathological behaviour.
        _log.warning(
            "chunk overlap >= chunk size; clamping",
            extra={"extra_fields": {"size": size, "overlap": overlap}},
        )
        overlap = max(0, size // 6)

    chunks: list[Chunk] = []
    for document in documents:
        chunks.extend(chunk_document(document, size=size, overlap=overlap))
    return chunks


def chunk_stats(chunks: Sequence[Chunk]) -> dict[str, float]:
    """Descriptive statistics, reported by ``/stats`` so the index is inspectable."""
    if not chunks:
        return {"count": 0.0, "mean_chars": 0.0, "min_chars": 0.0, "max_chars": 0.0}
    lengths = [len(chunk.text) for chunk in chunks]
    return {
        "count": float(len(chunks)),
        "mean_chars": round(sum(lengths) / len(lengths), 1),
        "min_chars": float(min(lengths)),
        "max_chars": float(max(lengths)),
    }
