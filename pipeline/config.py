"""Typed, cached application settings.

Two independent groups:

* :class:`Settings` -- ``MINING_*`` variables, shared by servers and client.
* :class:`LLMSettings` -- ``LLM_*`` variables, only relevant to the agent.

Both are cached via :func:`functools.lru_cache`; tests call the ``reset_*``
helpers to force a re-read after mutating ``os.environ``.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from pipeline.errors import ConfigurationError

__all__ = [
    "LLMSettings",
    "Settings",
    "get_llm_settings",
    "get_settings",
    "reset_settings_cache",
]

_DEFAULT_UA = (
    "mining-three-source-pipeline/0.1 (+https://github.com/example/mining-three-source-pipeline)"
)


class Settings(BaseSettings):
    """Runtime configuration shared across every component."""

    model_config = SettingsConfigDict(
        env_prefix="MINING_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    offline: bool = Field(
        default=True,
        description="When true, never touch the network; serve bundled fixtures.",
    )
    cache_dir: Path = Field(default=Path(".cache"), description="On-disk response cache root.")
    cache_ttl: int = Field(default=3600, ge=0, description="Cache entry lifetime in seconds.")
    cache_enabled: bool = Field(default=True, description="Disable to bypass the cache entirely.")

    http_timeout: float = Field(default=20.0, gt=0, description="Per-request timeout (seconds).")
    http_retries: int = Field(default=3, ge=1, le=10, description="Attempts per request.")
    user_agent: str = Field(default=_DEFAULT_UA)
    http_proxy: str | None = Field(default=None, description="Optional outbound proxy URL.")

    log_level: str = Field(default="INFO")
    log_json: bool = Field(default=False, description="Emit line-delimited JSON logs.")

    # -- ingest ---------------------------------------------------------------
    window_days: int = Field(
        default=30, ge=1, le=3650, description="Collection window; the brief asks for >= 30."
    )
    min_items_per_source: int = Field(
        default=200,
        ge=1,
        description="Minimum documents per source per run. The brief asks for >= 200.",
    )

    # -- indexing -------------------------------------------------------------
    index_dir: Path = Field(
        default=Path(".index"), description="Where the vector index and catalog live."
    )
    chunk_size: int = Field(default=900, ge=120, le=4000, description="Target chunk size (chars).")
    chunk_overlap: int = Field(
        default=150, ge=0, le=1000, description="Overlap between adjacent chunks (chars)."
    )
    embedder: str = Field(
        default="hashing",
        description="hashing (default, deterministic, offline) | openai",
    )
    embedding_dim: int = Field(
        default=512, ge=64, le=4096, description="Dimensions for the hashing embedder."
    )
    vector_backend: str = Field(
        default="numpy",
        description="numpy (default, exact cosine, no extra services) | chroma",
    )
    embedding_batch_size: int = Field(default=64, ge=1, le=512)

    # -- retrieval ------------------------------------------------------------
    top_k: int = Field(default=5, ge=1, le=50, description="Hits returned by /query.")
    candidate_k: int = Field(
        default=500,
        ge=1,
        le=100000,
        description=(
            "Cap on how many candidates get the (slower) lexical pass. Only applied "
            "once the corpus exceeds the store's pre-trim floor, so on the corpus "
            "size this brief targets it is inert."
        ),
    )
    lexical_weight: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Weight of the IDF-weighted lexical score mixed into the vector score. "
            "Left unset it is chosen from the embedder: a learned embedder leads with "
            "dense similarity, while the deterministic hashing embedder *is* a "
            "lexical retriever, so its precise half leads instead. Set it to pin an "
            "exact value."
        ),
    )
    # -- serving --------------------------------------------------------------
    serve_host: str = Field(
        default="127.0.0.1",
        description=(
            "Bind address for `mining-serve`. Loopback by default: this service has "
            "no authentication, and binding 0.0.0.0 by accident is how an internal "
            "demo becomes an open endpoint."
        ),
    )
    serve_port: int = Field(default=8000, ge=1, le=65535)
    allow_ingest_endpoint: bool = Field(
        default=False,
        description=(
            "Enable POST /ingest. Off by default because it mutates the index and "
            "fetches from the network; a read API that can be told to rewrite its own "
            "data is a surprising thing to expose."
        ),
    )
    title_weight: float = Field(
        default=0.15,
        ge=0.0,
        le=0.5,
        description=(
            "Extra weight for a match on the document title alone. A title is the "
            "most topical field a document has, and without a field boost a long "
            "document that mentions many concepts out-scores the short one whose "
            "title is exactly the question."
        ),
    )

    @field_validator("log_level")
    @classmethod
    def _upper_level(cls, value: str) -> str:
        level = value.upper()
        if level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError(f"unsupported log level: {value!r}")
        return level

    @field_validator("embedder", "vector_backend")
    @classmethod
    def _lower_choice(cls, value: str) -> str:
        return value.strip().lower()

    @field_validator("cache_dir", "index_dir", mode="before")
    @classmethod
    def _expand(cls, value: object) -> object:
        if isinstance(value, str):
            return Path(value).expanduser()
        return value


class LLMSettings(BaseSettings):
    """Configuration for the OpenAI-compatible endpoints used by answer synthesis.

    The same credential block serves two jobs -- chat completions for answers and
    (optionally) embeddings for retrieval. Both fall back to deterministic local
    implementations when ``api_key`` is empty, so the demo never depends on a
    paid key.
    """

    model_config = SettingsConfigDict(
        env_prefix="LLM_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    provider: str = Field(default="deepseek")
    model: str = Field(default="deepseek-chat")
    base_url: str = Field(default="https://api.deepseek.com/v1")
    api_key: str = Field(default="")
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, gt=0)

    embedding_model: str = Field(default="text-embedding-3-small")
    embedding_base_url: str = Field(
        default="https://api.openai.com/v1",
        description="Kept separate from ``base_url`` because chat and embedding "
        "endpoints are frequently different providers.",
    )
    embedding_api_key: str = Field(default="", description="Falls back to ``api_key`` when empty.")

    @property
    def chat_available(self) -> bool:
        """Whether a real chat completion can be attempted."""
        return bool(self.api_key and self.base_url)

    @property
    def embeddings_available(self) -> bool:
        """Whether a real embedding endpoint can be attempted."""
        return bool(self.embedding_key and self.embedding_base_url)

    @property
    def embedding_key(self) -> str:
        """The embedding credential, defaulting to the chat one."""
        return self.embedding_api_key or self.api_key

    def require_credentials(self) -> None:
        """Raise if this settings object cannot reach a real chat model."""
        if not self.api_key:
            raise ConfigurationError(
                "LLM_API_KEY is empty. Either set it, or leave "
                "MINING_OFFLINE=1 / omit the key to use the deterministic "
                "extractive answerer."
            )
        if not self.base_url:
            raise ConfigurationError("LLM_BASE_URL is empty")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide :class:`Settings` singleton."""
    return Settings()


@lru_cache(maxsize=1)
def get_llm_settings() -> LLMSettings:
    """Return the process-wide :class:`LLMSettings` singleton."""
    return LLMSettings()


def reset_settings_cache() -> None:
    """Drop cached settings so the next call re-reads the environment."""
    get_settings.cache_clear()
    get_llm_settings.cache_clear()
