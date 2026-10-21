"""Corpus statistics in one streaming pass.

    python -m findex.stats data/ [--limit N] [--top 50] [--eager]

The lazy pipeline is ``iter_documents -> tokenize -> Counter``: only the current
document and the term table are in memory. ``--eager`` runs the deliberately
list-based version for the eager-vs-lazy comparison in the README.
"""

from __future__ import annotations

import argparse
import logging
import resource
import sys
import time
import tracemalloc
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import islice
from pathlib import Path

from findex.corpus import Document, iter_documents
from findex.tokenize import tokenize


@dataclass
class CorpusStats:
    documents: int
    counts: Counter[str]  # the sink: the one structure that legitimately grows

    @property
    def tokens(self) -> int:
        return self.counts.total()

    @property
    def vocabulary(self) -> int:
        return len(self.counts)

    def top_terms(self, n: int = 50) -> list[tuple[str, int]]:
        return self.counts.most_common(n)


def corpus_stats(docs: Iterable[Document]) -> CorpusStats:
    """Lazy: consume a stream of documents, keep only the term counts."""
    counts: Counter[str] = Counter()
    documents = 0
    for doc in docs:
        documents += 1
        counts.update(tokenize(doc.text))
    return CorpusStats(documents, counts)


def corpus_stats_eager(root: Path, limit: int | None = None) -> CorpusStats:
    """Eager on purpose: every text, then every token list, is materialized first."""
    docs = list(islice(iter_documents(root), limit))
    token_lists = [list(tokenize(doc.text)) for doc in docs]
    counts: Counter[str] = Counter()
    for tokens in token_lists:
        counts.update(tokens)
    return CorpusStats(len(docs), counts)


def _format_bytes(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TiB"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m findex.stats", description=__doc__)
    parser.add_argument("root", type=Path, help="corpus directory or file")
    parser.add_argument("--limit", type=int, default=None, help="only the first N documents")
    parser.add_argument("--top", type=int, default=50, help="how many top terms to print")
    parser.add_argument("--eager", action="store_true", help="run the list-based version")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    tracemalloc.start()
    t0 = time.perf_counter()
    if args.eager:
        stats = corpus_stats_eager(args.root, args.limit)
    else:
        stats = corpus_stats(islice(iter_documents(args.root), args.limit))
    elapsed = time.perf_counter() - t0
    retained, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    max_rss *= 1 if sys.platform == "darwin" else 1024  # bytes on macOS, KiB on Linux

    print(f"mode:        {'eager (lists)' if args.eager else 'lazy (generators)'}")
    print(f"documents:   {stats.documents:,}")
    print(f"tokens:      {stats.tokens:,}")
    print(f"vocabulary:  {stats.vocabulary:,}")
    print(f"elapsed:     {elapsed:.2f} s")
    print(f"peak memory: {_format_bytes(peak)} (tracemalloc)")
    print(f"retained:    {_format_bytes(retained)} (still allocated at the end: the term table)")
    print(f"max RSS:     {_format_bytes(max_rss)} (whole process, incl. interpreter)")
    top_terms = stats.top_terms(args.top)
    print(f"\ntop {len(top_terms)} terms:")
    width = max((len(term) for term, _ in top_terms), default=0)
    for rank, (term, count) in enumerate(top_terms, start=1):
        print(f"{rank:>4}. {term:<{width}}  {count:>9,}")


if __name__ == "__main__":
    main()
