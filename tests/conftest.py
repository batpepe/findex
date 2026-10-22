"""Shared fixtures: a tiny on-disk corpus in every supported format, and its index."""

import json
from pathlib import Path

import pytest

from findex.corpus import iter_documents
from findex.index import Index, build_index
from findex.store import save

PEP_TEXT = """PEP: 9999
Title: The Walrus Guide
Author: Nobody <nobody@example.com>

Abstract
========

Assignment expressions use the walrus operator.
"""

FILES = {
    "docs/asyncio.txt": "Coroutines and tasks\n********************\n\nThe event loop runs tasks. "
    "An asyncio TaskGroup waits for every task.\n",
    "docs/threading.txt": "Threads\n*******\n\nThe GIL lets one thread run Python bytecode at a time.\n",
    "docs/unicode.md": "# Unicode HOWTO\n\nNormalize text with NFC before comparing: café == café.\n",
    "peps/pep-9999.rst": PEP_TEXT,
}
CRAWLED = [
    {"url": "https://example.org/a", "title": "Crawled A", "text": "The event loop of a crawler."},
    {"url": "https://example.org/b", "title": "Crawled B", "text": "Free-threaded Python has no GIL."},
]


@pytest.fixture(scope="session")
def corpus_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("corpus")
    for rel, text in FILES.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    with (root / "crawl.jsonl").open("w", encoding="utf-8") as f:
        for record in CRAWLED:
            f.write(json.dumps(record) + "\n")
    return root


@pytest.fixture(scope="session")
def corpus_index(corpus_dir: Path) -> Index:
    """Session-scoped and therefore shared: tests must treat it as read-only."""
    return build_index(iter_documents(corpus_dir), positions=True)


@pytest.fixture(scope="session")
def index_file(corpus_index: Index, tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("index") / "index.bin"
    save(corpus_index, path)
    return path
