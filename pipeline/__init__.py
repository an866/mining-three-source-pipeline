"""Mining news + policy + price pipeline -- shared core.

``pipeline`` is the lowest layer of the project: the collectors, cleaning,
de-duplication, embedding and vector store all live here, and ``serve`` / ``eval``
import from it. Nothing in this package imports from ``serve`` or ``eval``, so the
dependency direction is strictly one-way and there are no cycles.
"""

from pipeline.config import LLMSettings, Settings, get_llm_settings, get_settings
from pipeline.errors import (
    ConfigurationError,
    LLMError,
    OfflineModeError,
    ParseError,
    PipelineError,
    SourceUnavailableError,
)

__version__ = "0.1.0"

__all__ = [
    "ConfigurationError",
    "LLMError",
    "LLMSettings",
    "OfflineModeError",
    "ParseError",
    "PipelineError",
    "Settings",
    "SourceUnavailableError",
    "__version__",
    "get_llm_settings",
    "get_settings",
]
