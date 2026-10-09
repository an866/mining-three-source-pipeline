"""The HTTP surface: ``POST /query`` over the indexed corpus.

Run it::

    python -m pipeline.ingest          # build the index first
    mining-serve                       # or: python -m serve.app
    curl -s localhost:8000/query -H 'content-type: application/json' \
      -d '{"question":"印度尼西亚 2027 年镍矿配额调整了多少？"}' | jq

Design notes
------------
**Endpoints are sync ``def``, not ``async def``.** Retrieval is CPU-bound NumPy and
the embedding cache is blocking file I/O; declaring the handler ``async`` would run
that on the event loop and stall every other request. A sync handler runs in
Starlette's threadpool, and the store is read-only after load, so concurrent
``/query`` calls are safe without a lock. The only mutable state is the engine
reference, and ``/reload`` swaps it under a lock.

**A missing index is not a crash.** The service starts, ``/health`` reports
``index_loaded: false`` with the reason, and ``/query`` answers ``503`` carrying the
ingest command. Failing at import time would mean a container that crash-loops
instead of one whose health check tells you what to run.

**No authentication, bound to loopback.** See ``serve_host`` in
``pipeline/config.py``. ``POST /ingest`` is off unless
``MINING_ALLOW_INGEST_ENDPOINT=1``.
"""

from __future__ import annotations

import argparse
import threading
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from pipeline.answer import AnswerEngine
from pipeline.config import Settings, get_settings
from pipeline.errors import PipelineError
from pipeline.ingest import IndexNotBuiltError, IngestOutcome, ingest, load_index
from pipeline.llm import build_chat_client
from pipeline.logging_setup import configure_logging, get_logger
from pipeline.models import IndexStats, QueryAnswer, QueryFilters
from pipeline.retrieval import Retriever
from pipeline.vectorstore import VectorStore

__all__ = ["QueryRequest", "app", "main"]

_log = get_logger(__name__)

_SERVICE = "mining-three-source-pipeline"
_VERSION = "0.1.0"


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    """Body of ``POST /query``.

    ``extra="forbid"`` so a typo in a caller's payload (``"topk"``) is a 422 rather
    than a silently ignored field that returns five results when the caller wanted
    twenty.
    """

    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=2, max_length=1000)
    top_k: int | None = Field(default=None, ge=1, le=50)
    synthesise: bool = Field(
        default=True,
        description="Set false to force the extractive answer even when an LLM is configured.",
    )
    filters: QueryFilters | None = Field(
        default=None,
        description=(
            "Optional explicit constraints. Omit to let the planner infer them from "
            "the question; the applied filters are always echoed in the response."
        ),
    )


class ReloadResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reloaded: bool
    documents: int
    chunks: int
    took_ms: int


# ---------------------------------------------------------------------------
# Runtime state
# ---------------------------------------------------------------------------


class Runtime:
    """Holds the loaded engine and swaps it atomically on reload."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._lock = threading.Lock()
        self.engine: AnswerEngine | None = None
        self.store: VectorStore | None = None
        self.error: str | None = None
        self.answerer = "extractive"

    def load(self) -> None:
        """(Re)load the index and embedder. Never raises: reports into ``error``."""
        started = time.perf_counter()
        try:
            store, embedder = load_index(self.settings)
        except (IndexNotBuiltError, PipelineError) as exc:
            with self._lock:
                self.engine = None
                self.store = None
                self.error = str(exc)
            _log.error(
                "index unavailable; /query will return 503",
                extra={"extra_fields": {"error": str(exc)}},
            )
            return
        chat = build_chat_client(self.settings)
        engine = AnswerEngine(Retriever(store, embedder), chat, settings=self.settings)
        with self._lock:
            self.store = store
            self.engine = engine
            self.answerer = "llm" if chat is not None else "extractive"
            self.error = None
        stats = store.stats()
        _log.info(
            "index loaded",
            extra={
                "extra_fields": {
                    "chunks": stats.chunks,
                    "documents": stats.documents,
                    "backend": store.backend,
                    "answerer": self.answerer,
                    "took_ms": int((time.perf_counter() - started) * 1000),
                }
            },
        )

    def snapshot(self) -> tuple[AnswerEngine | None, VectorStore | None, str | None]:
        with self._lock:
            return self.engine, self.store, self.error


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the ASGI app. Exposed separately so tests can inject settings."""
    resolved = settings or get_settings()
    runtime = Runtime(resolved)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        configure_logging(resolved, force=True)
        runtime.load()
        yield

    app = FastAPI(
        title=_SERVICE,
        version=_VERSION,
        summary="Mining news + critical-minerals policy + price retrieval, in natural language.",
        lifespan=lifespan,
    )
    app.state.runtime = runtime

    @app.exception_handler(PipelineError)
    async def _pipeline_error(_: Request, exc: PipelineError) -> JSONResponse:
        """Domain errors become 502 with their own message, not a bare 500."""
        _log.error("pipeline error", extra={"extra_fields": {"error": str(exc)}})
        return JSONResponse(status_code=502, content={"detail": str(exc)})

    @app.get("/", include_in_schema=False)
    def index() -> dict[str, Any]:
        return {
            "service": _SERVICE,
            "version": _VERSION,
            "endpoints": ["/health", "/stats", "/query", "/reload"],
            "example": {
                "POST /query": {
                    "question": "印度尼西亚 2027 年镍矿配额调整了多少？",
                    "top_k": 5,
                }
            },
        }

    @app.get("/health")
    def health() -> dict[str, Any]:
        """Liveness plus whether the service can actually answer."""
        engine, store, error = runtime.snapshot()
        payload: dict[str, Any] = {
            "status": "ok" if engine is not None else "degraded",
            "index_loaded": engine is not None,
            "answerer": runtime.answerer,
            "offline": resolved.offline,
        }
        if store is not None:
            stats = store.stats()
            payload |= {
                "documents": stats.documents,
                "chunks": stats.chunks,
                "embedder": stats.embedder,
                "backend": stats.backend,
            }
        if error:
            payload["error"] = error
        return payload

    @app.get("/stats", response_model=IndexStats)
    def stats() -> IndexStats:
        _, store, error = runtime.snapshot()
        if store is None:
            raise HTTPException(status_code=503, detail=error or "index not loaded")
        return store.stats()

    @app.post("/query", response_model=QueryAnswer)
    def query(payload: QueryRequest) -> QueryAnswer:
        """Answer one natural-language question with citations."""
        engine, _, error = runtime.snapshot()
        if engine is None:
            raise HTTPException(status_code=503, detail=error or "index not loaded")
        return engine.answer(
            payload.question,
            top_k=payload.top_k,
            filters=payload.filters,
            synthesise=payload.synthesise,
        )

    @app.post("/reload", response_model=ReloadResponse)
    def reload() -> ReloadResponse:
        """Re-read the index from disk after a re-ingest, without a restart."""
        started = time.perf_counter()
        runtime.load()
        engine, store, _ = runtime.snapshot()
        if engine is None or store is None:
            raise HTTPException(status_code=503, detail="reload failed; index unavailable")
        stats = store.stats()
        return ReloadResponse(
            reloaded=True,
            documents=stats.documents,
            chunks=stats.chunks,
            took_ms=int((time.perf_counter() - started) * 1000),
        )

    @app.post("/ingest", response_model=None)
    def run_ingest(days: int | None = None) -> dict[str, Any]:
        """Re-collect and rebuild the index. Disabled unless explicitly enabled."""
        if not resolved.allow_ingest_endpoint:
            raise HTTPException(
                status_code=403,
                detail=(
                    "POST /ingest is disabled. Start the service with "
                    "MINING_ALLOW_INGEST_ENDPOINT=1 to enable it, or run "
                    "`python -m pipeline.ingest`."
                ),
            )
        outcome: IngestOutcome = ingest(resolved, days=days, index_dir=resolved.index_dir)
        runtime.load()
        return {
            "documents": outcome.stats.documents,
            "chunks": outcome.stats.chunks,
            "per_kind": outcome.stats.per_kind,
            "duplicates_removed": outcome.stats.duplicates_removed,
            "duplicates_by_stage": outcome.stats.duplicates_by_stage,
            "source_errors": outcome.stats.source_errors,
            "index_dir": str(outcome.index_dir) if outcome.index_dir else None,
        }

    return app


app = create_app()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mining-serve", description=__doc__)
    parser.add_argument("--host", default=None, help="Bind address (default MINING_SERVE_HOST).")
    parser.add_argument("--port", type=int, default=None, help="Port (default MINING_SERVE_PORT).")
    parser.add_argument("--reload", action="store_true", help="uvicorn autoreload (development).")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: ``mining-serve``."""
    import uvicorn

    args = _build_parser().parse_args(argv)
    settings = get_settings()
    host = args.host or settings.serve_host
    port = args.port or settings.serve_port
    _log.info("starting", extra={"extra_fields": {"host": host, "port": port}})
    if host not in ("127.0.0.1", "localhost", "::1"):
        _log.warning(
            "binding a non-loopback address; this service has no authentication",
            extra={"extra_fields": {"host": host}},
        )
    uvicorn.run(
        "serve.app:app" if args.reload else app,
        host=host,
        port=port,
        reload=args.reload,
        log_level="info",
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    import sys

    sys.exit(main())
