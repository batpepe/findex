"""findex - a full-text search engine over a corpus you choose."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("findex")
except PackageNotFoundError:  # running from a source tree that was never installed
    __version__ = "0.0.0"
