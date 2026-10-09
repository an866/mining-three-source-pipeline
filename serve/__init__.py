"""Serving package: the FastAPI application exposed by ``mining-serve``."""

from __future__ import annotations

__all__ = ["app", "create_app"]


def __getattr__(name: str) -> object:
    """Import the ASGI app lazily.

    ``serve/__init__.py`` importing ``serve.app`` eagerly would construct the app --
    and therefore read the settings file, the embedder and the logging config -- as a
    side effect of ``import serve``, which the test suite does to reach
    ``create_app``. Keeping it lazy means importing the package stays free.
    """
    if name in __all__:
        from serve import app as module

        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
