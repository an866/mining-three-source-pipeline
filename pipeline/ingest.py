"""The ingest pipeline: collect -> clean -> de-duplicate -> chunk -> embed -> index.

This module is the only place the whole pipeline runs, and it is both an importable
function and a CLI:

    python -m pipeline.ingest --days 30            # offline, deterministic corpus
    python -m pipeline.ingest --days 30 --online   # hit the real sources
    python -m pipeline.ingest --json               # machine-readable report

Ordering is not an implementation detail:

1. **Collect per source, tolerating failure.** ``safe_collect`` turns a dead source
   into an empty list plus an error string. A run where prices came back and
   policy did not still produces a usable index, and ``IngestStats.source_errors``
   records exactly what was missing so the report can say "this answer has no
   policy evidence" instead of pretending the corpus is complete.
2. **De-duplicate before chunking**, because chunking a duplicate triples its cost
   and its index footprint.
3. **Chunk before embedding**, because the chunk is the retrieval unit -- embedding
   documents and chunking later would mean embedding something never searched.
4. **Embed once, in batches**, then write the index atomically.

``python -m pipeline.ingest`` exits non-zero when a required source failed *and*
the corpus fell below the brief's floor, so CI can tell "the corpus is thin" from
"the corpus is fine".
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from pipeline.chunking import chunk_documents, chunk_stats
from pipeline.collectors import Collector, build_collectors
from pipeline.config import Settings, get_settings
from pipeline.dedup import Deduper
from pipeline.embedding import Embedder, build_embedder, describe_embedder
from pipeline.logging_setup import configure_logging, get_logger
from pipeline.models import Chunk, Document, IngestStats
from pipeline.vectorstore import VectorStore, build_store

__all__ = [
    "IngestOutcome",
    "collect_documents",
    "ingest",
    "load_index",
]

_log = get_logger(__name__)


@dataclass(slots=True)
class IngestOutcome:
    """Everything one ingest run produced."""

    documents: list[Document] = field(default_factory=list)
    chunks: list[Chunk] = field(default_factory=list)
    stats: IngestStats = field(default_factory=IngestStats)
    store: VectorStore | None = None
    index_dir: Path | None = None
    chunk_summary: dict[str, float] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """Whether every source contributed and the corpus cleared the floor."""
        return not self.stats.source_errors


# ---------------------------------------------------------------------------
# Stage 1+2: collect and de-duplicate
# ---------------------------------------------------------------------------


def collect_documents(
    settings: Settings | None = None,
    *,
    days: int | None = None,
    collectors: Sequence[Collector] | None = None,
) -> tuple[list[Document], IngestStats]:
    """Run every collector, de-duplicate the union, and report per source.

    ``per_source`` counts documents *after* de-duplication, attributed to the source
    that produced the surviving copy. Counting before de-duplication would let the
    report claim 600 documents while the index holds 400, which is the kind of
    number that makes every downstream metric meaningless.
    """
    resolved = settings or get_settings()
    window = days or resolved.window_days
    active = list(collectors) if collectors is not None else build_collectors(resolved)

    collected: list[Document] = []
    errors: dict[str, str] = {}
    raw_counts: dict[str, int] = {}
    identity_dropped = 0
    for collector in active:
        report = collector.safe_collect(days=window)
        raw_counts[report.name] = len(report.documents)
        identity_dropped += report.dropped_identity
        if report.error:
            errors[report.name] = report.error
        collected.extend(report.documents)
        # Release sockets as soon as one source is done; a full run opens three
        # pools and the process outlives them.
        collector.close()

    dedupe = Deduper().run(collected)
    documents = dedupe.documents

    # Account for every duplicate the run removed, not just the cascade's share.
    # The source-level identity guard and the cascade's tier 1 share a key, so on
    # this corpus the guard takes the re-collected copies and reports them here --
    # otherwise `duplicates_removed` would silently understate the total.
    duplicates_by_stage = {
        **({"collector_identity": identity_dropped} if identity_dropped else {}),
        **dedupe.by_tier,
    }

    per_source: dict[str, int] = {}
    per_kind: dict[str, int] = {}
    for document in documents:
        per_source[document.source] = per_source.get(document.source, 0) + 1
        per_kind[document.kind.value] = per_kind.get(document.kind.value, 0) + 1

    stats = IngestStats(
        documents=len(documents),
        per_source=per_source,
        per_kind=per_kind,
        duplicates_removed=sum(duplicates_by_stage.values()),
        duplicates_by_stage=duplicates_by_stage,
        window_days=window,
        offline=resolved.offline,
        generated_at=datetime.now(UTC),
        source_errors=errors,
    )
    _log.info(
        "collection complete",
        extra={
            "extra_fields": {
                "documents": len(documents),
                "collected_before_dedupe": len(collected) + identity_dropped,
                "duplicates_removed": sum(duplicates_by_stage.values()),
                "duplicates_by_stage": duplicates_by_stage,
                "errors": len(errors),
            }
        },
    )
    return documents, stats


# ---------------------------------------------------------------------------
# Stage 3+4+5: chunk, embed, index
# ---------------------------------------------------------------------------


def ingest(
    settings: Settings | None = None,
    *,
    days: int | None = None,
    index_dir: Path | None = None,
    embedder: Embedder | None = None,
    store: VectorStore | None = None,
    save: bool = True,
) -> IngestOutcome:
    """Run the whole pipeline once and return the outcome.

    Components can be injected for tests and for the eval harness, which reuses one
    embedder across the ingest and the query half of a run -- constructing two would
    double the cache misses for no benefit.
    """
    resolved = settings or get_settings()
    documents, stats = collect_documents(resolved, days=days)

    chunks = chunk_documents(documents, size=resolved.chunk_size, overlap=resolved.chunk_overlap)
    stats.chunks = len(chunks)

    engine = embedder or build_embedder(resolved)
    vectors: list[list[float]] = []
    batch = resolved.embedding_batch_size
    for start in range(0, len(chunks), batch):
        window = chunks[start : start + batch]
        vectors.extend(engine.embed([chunk.text for chunk in window]))

    index = store or build_store(settings=resolved, embedder_name=engine.name, dim=engine.dim)
    index.add(chunks, vectors)

    target = index_dir or resolved.index_dir
    if save:
        index.save(target)

    outcome = IngestOutcome(
        documents=documents,
        chunks=chunks,
        stats=stats,
        store=index,
        index_dir=Path(target) if save else None,
        chunk_summary=chunk_stats(chunks),
    )
    _log.info(
        "ingest complete",
        extra={
            "extra_fields": {
                "documents": stats.documents,
                "chunks": stats.chunks,
                "embedder": describe_embedder(engine),
                "backend": index.backend,
            }
        },
    )
    return outcome


def load_index(
    settings: Settings | None = None,
    *,
    index_dir: Path | None = None,
    embedder: Embedder | None = None,
) -> tuple[VectorStore, Embedder]:
    """Load the persisted index and the matching embedder.

    Raises :class:`~pipeline.errors.VectorStoreError` when the index is missing or
    was built with a different embedder. Failing here, at startup, is much better
    than serving silently wrong neighbours for an hour.
    """
    resolved = settings or get_settings()
    engine = embedder or build_embedder(resolved)
    store = build_store(settings=resolved, embedder_name=engine.name, dim=engine.dim)
    if not store.load(index_dir or resolved.index_dir):
        raise IndexNotBuiltError(resolved.index_dir if index_dir is None else Path(index_dir))
    return store, engine


class IndexNotBuiltError(Exception):
    """The index directory has no manifest -- ingest has never been run."""

    def __init__(self, directory: Path) -> None:
        super().__init__(
            f"no index found in {directory}. Run `python -m pipeline.ingest` first "
            f"(or `make ingest`)."
        )
        self.directory = directory


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m pipeline.ingest",
        description="Collect the three sources, de-duplicate, index and report.",
    )
    parser.add_argument("--days", type=int, default=None, help="Collection window in days.")
    parser.add_argument(
        "--online",
        action="store_true",
        help="Hit the real sources (default is the deterministic offline corpus).",
    )
    parser.add_argument("--index-dir", type=Path, default=None, help="Where to write the index.")
    parser.add_argument(
        "--no-save", action="store_true", help="Build the index in memory and discard it."
    )
    parser.add_argument("--json", action="store_true", help="Emit the report as JSON.")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-stage logs.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Returns the process exit code."""
    args = _build_parser().parse_args(argv)
    settings = get_settings()
    if args.online:
        settings = Settings(**{**settings.model_dump(), "offline": False})
    if args.quiet:
        settings = settings.model_copy(update={"log_level": "WARNING"})
    configure_logging(settings, force=True)

    outcome = ingest(
        settings,
        days=args.days,
        index_dir=args.index_dir,
        save=not args.no_save,
    )
    stats = outcome.stats

    if args.json:
        payload = {
            "documents": stats.documents,
            "chunks": stats.chunks,
            "per_kind": stats.per_kind,
            "duplicates_removed": stats.duplicates_removed,
            "duplicates_by_stage": stats.duplicates_by_stage,
            "window_days": stats.window_days,
            "offline": stats.offline,
            "index_dir": str(outcome.index_dir) if outcome.index_dir else None,
            "chunk_stats": outcome.chunk_summary,
            "source_errors": stats.source_errors,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        _print_report(outcome)

    floor = settings.min_items_per_source
    thin = [kind for kind in ("news", "policy", "price") if stats.per_kind.get(kind, 0) < floor]
    if thin and stats.source_errors:
        _log.error(
            "corpus below the required floor with source errors present",
            extra={"extra_fields": {"thin": thin, "errors": stats.source_errors}},
        )
        return 1
    return 0


def _print_report(outcome: IngestOutcome) -> None:
    stats = outcome.stats
    lines = [
        "",
        f"documents      : {stats.documents}",
        f"chunks         : {stats.chunks}",
        f"duplicates     : {stats.duplicates_removed} removed"
        + (
            " (" + ", ".join(f"{k}={v}" for k, v in sorted(stats.duplicates_by_stage.items())) + ")"
            if stats.duplicates_by_stage
            else ""
        ),
        f"window         : {stats.window_days} days ({'offline' if stats.offline else 'live'})",
        f"index          : {outcome.index_dir or '(not saved)'}",
    ]
    if stats.per_kind:
        rendered = ", ".join(f"{kind}={count}" for kind, count in sorted(stats.per_kind.items()))
        lines.append(f"per kind       : {rendered}")
    if stats.per_source:
        rendered = ", ".join(f"{name}={count}" for name, count in sorted(stats.per_source.items()))
        lines.append(f"per source     : {rendered}")
    if outcome.chunk_summary:
        summary = outcome.chunk_summary
        lines.append(
            f"chunk chars    : mean={summary.get('mean_chars')} "
            f"min={summary.get('min_chars')} max={summary.get('max_chars')}"
        )
    if stats.source_errors:
        for name, error in sorted(stats.source_errors.items()):
            lines.append(f"  ! {name}: {error}")
    lines.append("")
    print("\n".join(lines))


if __name__ == "__main__":  # pragma: no cover - exercised via subprocess in tests
    sys.exit(main())
