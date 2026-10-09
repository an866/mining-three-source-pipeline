"""Embedders: text -> dense vector, with a deterministic default.

Two implementations, one interface:

* :class:`HashingEmbedder` -- the default. Signed feature hashing over the
  tokeniser's output (Latin words + CJK unigrams and bigrams), plus the bilingual
  concept markers from :func:`pipeline.textutils.concepts_in`, sublinear term
  frequency, L2 normalised. Deterministic across processes and machines, needs no
  network and no key, and is a genuinely usable lexical-semantic retriever at this
  corpus size. It is not a learned embedding and does not claim to be; DATA_NOTES.md
  and ``/stats`` both say so.
* :class:`OpenAIEmbedder` -- any OpenAI-compatible ``/embeddings`` endpoint, so the
  same code path serves OpenAI, DeepSeek, Voyage, a local vLLM or an Ollama
  gateway. Selected with ``MINING_EMBEDDER=openai`` plus ``LLM_EMBEDDING_*``.

Why feature hashing is a defensible default: the alternative is a demo that does
not run without a paid key. Recall@5 measured against a deterministic hash
embedder is a *lower bound* on what a learned embedder achieves, and — more
usefully — it is reproducible, so a regression in the retriever is visible instead
of being masked by embedding noise.

:class:`CachedEmbedder` wraps either one with the on-disk cache, keyed on
``(embedder name, dimension, sha256(text))``. Re-running ingest therefore costs
nothing for unchanged chunks, which is what makes the "delete the index and
rebuild" workflow in RUN.md practical.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import httpx

from pipeline.cache import DiskCache
from pipeline.config import LLMSettings, Settings, get_llm_settings, get_settings
from pipeline.errors import ConfigurationError, LLMError
from pipeline.logging_setup import get_logger
from pipeline.textutils import concepts_in, normalize_whitespace, tokenize

__all__ = [
    "CachedEmbedder",
    "Embedder",
    "HashingEmbedder",
    "OpenAIEmbedder",
    "build_embedder",
]

_log = get_logger(__name__)

_CACHE_NAMESPACE = "embeddings"


@runtime_checkable
class Embedder(Protocol):
    """The contract every embedder satisfies."""

    #: Stable identifier; part of the cache key and reported by ``/stats``.
    name: str
    #: Vector length. Must not change without rebuilding the index.
    dim: int

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of documents."""

    def embed_query(self, text: str) -> list[float]:
        """Embed one query.

        Separate from :meth:`embed` on purpose: asymmetric retrievers (E5-style
        ``query:``/``passage:`` prefixes) need the distinction, and having one
        method would make that a breaking change later.
        """


class HashingEmbedder:
    """Deterministic signed feature hashing.

    ``dim=512`` is the default. Larger dimensions reduce collisions; past ~1024 on
    a corpus of this size the returns are not measurable, and the vector store
    holds everything in RAM.
    """

    name = "hashing"

    def __init__(self, *, dim: int = 512) -> None:
        if dim < 16:
            raise ConfigurationError(f"embedding dimension too small: {dim}")
        self.dim = dim

    # -- Embedder -----------------------------------------------------------

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    # -- internals ----------------------------------------------------------

    def _vector(self, text: str) -> list[float]:
        tokens = tokenize(text or "")
        concepts = concepts_in(text or "")
        if not tokens and not concepts:
            # A zero vector would be indistinguishable from "no match" and would
            # produce NaN under normalisation; a unit vector in a fixed slot at
            # least ranks consistently.
            vector = [0.0] * self.dim
            vector[0] = 1.0
            return vector

        counts = Counter(tokens)
        # Bigrams over *tokens*, not characters: the tokeniser already expanded
        # CJK runs into unigrams and bigrams, so this adds word-order signal for
        # Latin text without double-counting Chinese.
        bigrams = Counter(f"{tokens[i]}\u241f{tokens[i + 1]}" for i in range(len(tokens) - 1))

        vector = [0.0] * self.dim
        for token, frequency in counts.items():
            index, sign = self._bucket(token, salt="tok")
            # Sublinear tf: the tenth mention of "lithium" says much less than the
            # second, and raw counts let long documents dominate.
            vector[index] += sign * (1.0 + math.log(frequency))
        for bigram, frequency in bigrams.items():
            index, sign = self._bucket(bigram, salt="bi")
            vector[index] += sign * 0.5 * (1.0 + math.log(frequency))
        # Concepts carry the cross-lingual signal and are weighted above a plain
        # token: 稀土 and "rare earth" both hash to ``concept:rare_earth``, which is
        # what lets a Chinese question retrieve an English policy document. The
        # weight is a flat 2.0 rather than a tuned constant because there is no
        # held-out data to tune against, and a flat weight is at least honest about
        # being a prior.
        for concept in concepts:
            index, sign = self._bucket(f"concept:{concept}", salt="con")
            vector[index] += sign * 2.0

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            vector[0] = 1.0
            return vector
        return [value / norm for value in vector]

    def _bucket(self, token: str, *, salt: str) -> tuple[int, float]:
        """Hash a token to a dimension and a sign.

        The sign matters: without it, unrelated tokens colliding on one dimension
        always *add*, inflating similarity between documents that share nothing.
        A signed hash makes collisions cancel in expectation.

        ``blake2b`` rather than the builtin ``hash()``, which is salted per process
        and would make two runs over the same text produce different vectors -- and
        therefore different eval scores.
        """
        digest = hashlib.blake2b(f"{salt}|{token}".encode(), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        return value % self.dim, 1.0 if value & 1 else -1.0


class OpenAIEmbedder:
    """Any OpenAI-compatible ``/embeddings`` endpoint.

    Deliberately not using the ``openai`` SDK: one HTTP call needs ``httpx``, which
    is already a dependency, and avoiding the SDK keeps the failure modes visible
    (a bad base URL and an expired key produce different HTTP statuses, and both
    surface as typed errors rather than an opaque SDK exception).
    """

    name = "openai"

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str,
        dim: int,
        settings: Settings | None = None,
        timeout: float = 60.0,
    ) -> None:
        if not api_key:
            raise ConfigurationError(
                "LLM_EMBEDDING_API_KEY (or LLM_API_KEY) is required when MINING_EMBEDDER=openai"
            )
        self.model = model
        self.dim = dim
        self._url = base_url.rstrip("/") + "/embeddings"
        self._settings = settings or get_settings()
        self._client = httpx.Client(
            timeout=httpx.Timeout(timeout),
            headers={"Authorization": f"Bearer {api_key}"},
        )

    def close(self) -> None:
        self._client.close()

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        payload = {"model": self.model, "input": [normalize_whitespace(t) for t in texts]}
        try:
            response = self._client.post(self._url, json=payload)
        except httpx.HTTPError as exc:
            raise LLMError(f"embedding request failed: {exc}") from exc
        if response.status_code >= 400:
            raise LLMError(
                f"embedding endpoint returned HTTP {response.status_code}: {response.text[:200]}"
            )
        try:
            body = response.json()
            rows = sorted(body["data"], key=lambda row: row.get("index", 0))
            vectors = [list(map(float, row["embedding"])) for row in rows]
        except (KeyError, TypeError, ValueError) as exc:
            raise LLMError(f"unexpected embedding response shape: {exc}") from exc
        if len(vectors) != len(texts):
            raise LLMError(
                f"embedding endpoint returned {len(vectors)} vectors for {len(texts)} inputs"
            )
        for vector in vectors:
            if len(vector) != self.dim:
                raise LLMError(
                    f"embedding dimension mismatch: configured {self.dim}, got {len(vector)}"
                )
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]


class CachedEmbedder:
    """Memoises an embedder onto the disk cache.

    Keyed on the embedder's identity, its dimension and the exact text, so a change
    to any of the three misses cleanly instead of returning a stale vector of the
    wrong shape. Batching is preserved: only misses are sent upstream.
    """

    def __init__(self, inner: Embedder, cache: DiskCache | None = None) -> None:
        self._inner = inner
        self._cache = cache if cache is not None else DiskCache()
        self.name = inner.name
        self.dim = inner.dim

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        resolved: list[list[float] | None] = []
        misses: list[int] = []
        for index, text in enumerate(texts):
            cached = self._cache.get(_CACHE_NAMESPACE, self._key(text))
            if isinstance(cached, list) and len(cached) == self.dim:
                resolved.append([float(value) for value in cached])
            else:
                resolved.append(None)
                misses.append(index)

        if misses:
            produced = self._inner.embed([texts[index] for index in misses])
            for offset, index in enumerate(misses):
                vector = produced[offset]
                resolved[index] = vector
                self._cache.set(_CACHE_NAMESPACE, self._key(texts[index]), vector)
            _log.info(
                "embedded batch",
                extra={
                    "extra_fields": {
                        "embedder": self.name,
                        "hits": len(texts) - len(misses),
                        "misses": len(misses),
                    }
                },
            )

        return [vector or self._zero() for vector in resolved]

    def embed_query(self, text: str) -> list[float]:
        # Queries are cached too: the eval harness asks the same questions
        # repeatedly while iterating on ranking, and each ask would otherwise be a
        # paid round trip.
        cached = self._cache.get(_CACHE_NAMESPACE, self._key(text))
        if isinstance(cached, list) and len(cached) == self.dim:
            return [float(value) for value in cached]
        vector = self._inner.embed_query(text)
        self._cache.set(_CACHE_NAMESPACE, self._key(text), vector)
        return vector

    def _key(self, text: str) -> str:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return f"{self.name}:{self.dim}:{digest}"

    def _zero(self) -> list[float]:  # pragma: no cover - defensive
        return [0.0] * self.dim


def build_embedder(
    settings: Settings | None = None,
    llm_settings: LLMSettings | None = None,
    *,
    use_cache: bool = True,
) -> Embedder:
    """Construct the configured embedder.

    ``MINING_EMBEDDER=hashing`` (default) never fails. ``=openai`` requires a key
    and raises :class:`ConfigurationError` at construction if one is missing --
    failing at startup is much better than failing on the first query.
    """
    resolved = settings or get_settings()
    choice = resolved.embedder

    inner: Embedder
    if choice == "hashing":
        inner = HashingEmbedder(dim=resolved.embedding_dim)
    elif choice == "openai":
        llm = llm_settings or get_llm_settings()
        if not llm.embeddings_available:
            raise ConfigurationError(
                "MINING_EMBEDDER=openai requires LLM_EMBEDDING_API_KEY or LLM_API_KEY"
            )
        inner = OpenAIEmbedder(
            model=llm.embedding_model,
            api_key=llm.embedding_key,
            base_url=llm.embedding_base_url,
            dim=resolved.embedding_dim,
            settings=resolved,
        )
    else:
        raise ConfigurationError(
            f"unknown MINING_EMBEDDER={choice!r}; expected 'hashing' or 'openai'"
        )

    if not use_cache:
        return inner
    return CachedEmbedder(inner, DiskCache(settings=resolved))


def describe_embedder(embedder: Embedder) -> dict[str, object]:
    """A JSON-safe description, embedded in ``/stats`` and in the eval report."""
    return {"name": embedder.name, "dim": embedder.dim}
