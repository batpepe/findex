"""The inverted index: term -> postings sorted by doc_id.

    python -m findex.index data/ --out index.bin [--positions] [--format binary|pickle] [--limit N]

Built in one pass over the lazily streamed corpus: for each document a local
``Counter`` (or position lists) is filled, then *flushed* into the global
postings when the document ends. Documents arrive in increasing ``doc_id``
order, so appending keeps every postings list sorted without a sort.
"""

from __future__ import annotations

import argparse
import logging
import time
import tracemalloc
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path

from findex.corpus import Document, iter_documents
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


@dataclass
class InvertedIndex:
    postings: Mapping[str, list[Posting]] = field(default_factory=dict)  # dict, or lazy when loaded
    doc_lengths: dict[int, int] = field(default_factory=dict)  # tokens per document
    doc_meta: dict[int, DocMeta] = field(default_factory=dict)
    has_positions: bool = False

    @property
    def num_docs(self) -> int:
        return len(self.doc_lengths)

    def num_postings(self) -> int:
        return sum(len(plist) for plist in self.postings.values())


def build_index(docs: Iterable[Document], *, positions: bool = False) -> InvertedIndex:
    """Single streaming pass; only the index grows, the corpus never sits in memory."""
    postings: defaultdict[str, list[Posting]] = defaultdict(list)
    index = InvertedIndex(has_positions=positions)
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
        index.doc_lengths[doc.doc_id] = length
        index.doc_meta[doc.doc_id] = DocMeta(doc.path, doc.title)
    index.postings = dict(postings)  # freeze: a missing term must not silently create an entry
    return index


def main(argv: list[str] | None = None) -> None:
    from findex.store import save  # imported here: store itself imports this module

    parser = argparse.ArgumentParser(prog="python -m findex.index", description=__doc__)
    parser.add_argument("root", type=Path, help="corpus directory or file")
    parser.add_argument("--out", type=Path, required=True, help="where to write the index")
    parser.add_argument("--format", choices=("binary", "pickle"), default="binary")
    parser.add_argument("--positions", action="store_true", help="store token positions (phrases)")
    parser.add_argument("--limit", type=int, default=None, help="only the first N documents")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    tracemalloc.start()
    t0 = time.perf_counter()
    index = build_index(islice(iter_documents(args.root), args.limit), positions=args.positions)
    built = time.perf_counter()
    save(index, args.out, fmt=args.format)
    saved = time.perf_counter()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"documents:   {index.num_docs:,}")
    print(f"terms:       {len(index.postings):,}")
    print(f"postings:    {index.num_postings():,}")
    print(f"positions:   {'yes' if index.has_positions else 'no'}")
    print(f"build:       {built - t0:.2f} s")
    print(f"save:        {saved - built:.2f} s ({args.format}, {args.out.stat().st_size / 2**20:.1f} MiB)")
    print(f"peak memory: {peak / 2**20:.1f} MiB (tracemalloc)")


if __name__ == "__main__":
    main()
