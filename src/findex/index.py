"""The inverted index: term -> postings sorted by doc_id.

    python -m findex.index data/ --out index.bin [--positions] [--format binary|pickle] [--limit N]

Built in one pass over the lazily streamed corpus: for each document a local
``Counter`` (or position lists) is filled, then *flushed* into the global
postings when the document ends. Documents arrive in increasing ``doc_id``
order, so appending keeps every postings list sorted without a sort.

``Index`` speaks the Python data model: it is a read-only
``Mapping[str, list[Posting]]`` (``len``, ``in``, ``[]``, iteration come from
``collections.abc``), with corpus statistics as properties.
"""

from __future__ import annotations

import argparse
import functools
import logging
import time
import tracemalloc
import zlib
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Protocol

from findex.corpus import Document, iter_documents
from findex.timing import timed
from findex.tokenize import tokenize

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Posting:
    doc_id: int
    tf: int  # term frequency: occurrences of the term in this document
    positions: tuple[int, ...] = ()  # token offsets, only with --positions


@dataclass(frozen=True, slots=True)
class DocMeta:
    path: str
    title: str


class DocStore(Protocol):
    """Where the original texts live, so results can show snippets without the corpus."""

    def text(self, doc_id: int) -> str: ...


class CompressedTexts:
    """Document texts kept zlib-compressed in memory (~3.5x smaller than the text)."""

    def __init__(self, blobs: dict[int, bytes] | None = None) -> None:
        self.blobs: dict[int, bytes] = blobs if blobs is not None else {}

    def add(self, doc_id: int, text: str) -> None:
        self.blobs[doc_id] = zlib.compress(text.encode(), 6)

    def text(self, doc_id: int) -> str:
        return zlib.decompress(self.blobs[doc_id]).decode()


class Index(Mapping[str, list[Posting]]):
    """A read-only inverted index that behaves like a Python mapping."""

    def __init__(
        self,
        postings: Mapping[str, list[Posting]],
        doc_lengths: dict[int, int],
        doc_meta: dict[int, DocMeta],
        *,
        has_positions: bool = False,
        texts: DocStore | None = None,
    ) -> None:
        self._postings = postings
        self.doc_lengths = doc_lengths
        self.doc_meta = doc_meta
        self.has_positions = has_positions
        self.texts = texts
        self._closers: list[Callable[[], None]] = []

    # --- the Mapping protocol: these three give us keys/items/values/get/==/in for free ---

    def __getitem__(self, term: str) -> list[Posting]:
        return self._postings[term]  # KeyError if the term was never indexed

    def __iter__(self) -> Iterator[str]:
        return iter(self._postings)

    def __len__(self) -> int:
        return len(self._postings)  # number of distinct terms

    def __contains__(self, term: object) -> bool:  # faster than Mapping's try/except default
        return term in self._postings

    def __repr__(self) -> str:
        name = type(self).__name__
        return f"{name}(terms={len(self):_}, docs={self.num_docs:_}, positions={self.has_positions})"

    @property
    def postings(self) -> Mapping[str, list[Posting]]:
        """The underlying term -> postings mapping (a dict, or lazy mmap-backed after loading)."""
        return self._postings

    # --- corpus statistics ---

    @property
    def num_docs(self) -> int:
        return len(self.doc_lengths)

    @functools.cached_property
    def avg_doc_length(self) -> float:  # BM25 needs it for every scored posting: compute once
        return sum(self.doc_lengths.values()) / self.num_docs if self.num_docs else 0.0

    @functools.cached_property
    def all_doc_ids(self) -> frozenset[int]:
        return frozenset(self.doc_lengths)

    def doc_length(self, doc_id: int) -> int:
        return self.doc_lengths[doc_id]

    def df(self, term: str) -> int:
        """Document frequency: in how many documents the term occurs."""
        df = getattr(self._postings, "df", None)  # the mmap-backed postings know it without decoding
        if df is not None:
            return df(term) if term in self._postings else 0
        return len(self._postings.get(term, ()))

    def title(self, doc_id: int) -> str:
        return self.doc_meta[doc_id].title

    def text(self, doc_id: int) -> str:
        if self.texts is None:
            raise LookupError("this index was saved without document texts")
        return self.texts.text(doc_id)

    def num_postings(self) -> int:
        return sum(self.df(term) for term in self)

    # --- lifetime ---

    def on_close(self, closer: Callable[[], None]) -> None:
        self._closers.append(closer)

    def close(self) -> None:
        while self._closers:
            self._closers.pop()()


@timed
def build_index(docs: Iterable[Document], *, positions: bool = False, store_texts: bool = True) -> Index:
    """Single streaming pass; only the index grows, the corpus never sits in memory."""
    postings: defaultdict[str, list[Posting]] = defaultdict(list)
    doc_lengths: dict[int, int] = {}
    doc_meta: dict[int, DocMeta] = {}
    texts = CompressedTexts()
    for doc in docs:
        if positions:
            term_positions: defaultdict[str, list[int]] = defaultdict(list)
            position = -1
            for position, token in enumerate(tokenize(doc.text)):
                term_positions[token].append(position)
            length = position + 1
            for term, offsets in term_positions.items():
                postings[term].append(Posting(doc.doc_id, len(offsets), tuple(offsets)))
        else:
            counts = Counter(tokenize(doc.text))
            length = counts.total()
            for term, tf in counts.items():
                postings[term].append(Posting(doc.doc_id, tf))
        doc_lengths[doc.doc_id] = length
        doc_meta[doc.doc_id] = DocMeta(doc.path, doc.title)
        if store_texts:
            texts.add(doc.doc_id, doc.text)
    # dict(): a missing term must raise KeyError, not silently create an empty list
    return Index(
        dict(postings), doc_lengths, doc_meta, has_positions=positions, texts=texts if store_texts else None
    )


@contextmanager
def open_index(path: Path) -> Iterator[Index]:
    """``with open_index(p) as ix:`` - the memory map is released even if the body raises."""
    from findex.store import load  # imported here: store itself imports this module

    index = load(path)
    try:
        yield index
    finally:
        index.close()


def main(argv: list[str] | None = None) -> None:
    from findex.store import save

    parser = argparse.ArgumentParser(prog="python -m findex.index", description=__doc__)
    parser.add_argument("root", type=Path, help="corpus directory or file")
    parser.add_argument("--out", type=Path, required=True, help="where to write the index")
    parser.add_argument("--format", choices=("binary", "pickle"), default="binary")
    parser.add_argument("--positions", action="store_true", help="store token positions (phrases)")
    parser.add_argument("--limit", type=int, default=None, help="only the first N documents")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging (timings)")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )

    tracemalloc.start()
    t0 = time.perf_counter()
    index = build_index(islice(iter_documents(args.root), args.limit), positions=args.positions)
    built = time.perf_counter()
    save(index, args.out, fmt=args.format)
    saved = time.perf_counter()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(repr(index))
    print(f"documents:   {index.num_docs:,}")
    print(f"terms:       {len(index):,}")
    print(f"postings:    {index.num_postings():,}")
    print(f"positions:   {'yes' if index.has_positions else 'no'}")
    print(f"build:       {built - t0:.2f} s")
    print(f"save:        {saved - built:.2f} s ({args.format}, {args.out.stat().st_size / 2**20:.1f} MiB)")
    print(f"peak memory: {peak / 2**20:.1f} MiB (tracemalloc)")


if __name__ == "__main__":
    main()
