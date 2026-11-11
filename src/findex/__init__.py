"""findex - a full-text search engine over a corpus you choose."""

from __future__ import annotations

import functools


def __getattr__(name: str) -> str:
    # PEP 562: ``findex.__version__`` is computed on first use. importlib.metadata costs
    # ~20 ms to import, and only `--version` and the web app ever ask (Lab 8).
    if name == "__version__":
        return _version()
    raise AttributeError(f"module 'findex' has no attribute {name!r}")


@functools.cache
def _version() -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("findex")
    except PackageNotFoundError:  # running from a source tree that was never installed
        return "0.0.0"
