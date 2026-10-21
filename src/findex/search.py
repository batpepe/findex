"""Boolean search over the inverted index.

    python -m findex.search index.bin "query" [--engine merge|set] [--limit 10]

Query language (Lab 2 - Lab 3 replaces it with a real parser): words separated
by spaces are ANDed; ``OR`` and ``NOT`` are evaluated strictly left to right,
with no precedence and no parentheses yet::

    asyncio task        ->  asyncio AND task
    thread OR process   ->  thread OR process
    python NOT java     ->  python AND NOT java
    a OR b c            ->  (a OR b) AND c

Query words go through the same tokenizer as documents, so ``Python's`` finds
``python`` and ``free-threaded`` means ``free AND threaded``.
"""

from __future__ import annotations

import argparse
import time
import tracemalloc
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal

from findex.index import InvertedIndex
from findex.store import load
from findex.tokenize import tokenize

type Engine = Literal["merge", "set"]
type SetOp = Callable[[Sequence[int], Sequence[int]], list[int]]

OPERATORS = frozenset({"AND", "OR", "NOT"})


# --- two-pointer merges over sorted doc-id lists: O(len(a) + len(b)), no hashing ---


def intersect(a: Sequence[int], b: Sequence[int]) -> list[int]:
    out: list[int] = []
    i = j = 0
    while i < len(a) and j < len(b):
        x, y = a[i], b[j]
        if x == y:
            out.append(x)
            i += 1
            j += 1
        elif x < y:
            i += 1
        else:
            j += 1
    return out


def union(a: Sequence[int], b: Sequence[int]) -> list[int]:
    out: list[int] = []
    i = j = 0
    while i < len(a) and j < len(b):
        x, y = a[i], b[j]
        if x == y:
            out.append(x)
            i += 1
            j += 1
        elif x < y:
            out.append(x)
            i += 1
        else:
            out.append(y)
            j += 1
    out.extend(a[i:])
    out.extend(b[j:])
    return out


def difference(a: Sequence[int], b: Sequence[int]) -> list[int]:
    """``a AND NOT b``: keep the ids of ``a`` that the merge does not find in ``b``."""
    out: list[int] = []
    i = j = 0
    while i < len(a):
        if j == len(b) or a[i] < b[j]:
            out.append(a[i])
            i += 1
        elif a[i] == b[j]:
            i += 1
            j += 1
        else:
            j += 1
    return out


MERGE_OPS: dict[str, SetOp] = {"AND": intersect, "OR": union, "NOT": difference}
SET_OPS: dict[str, SetOp] = {
    "AND": lambda a, b: sorted(set(a).intersection(b)),
    "OR": lambda a, b: sorted(set(a).union(b)),
    "NOT": lambda a, b: sorted(set(a).difference(b)),
}


def doc_ids(index: InvertedIndex, term: str) -> list[int]:
    return [p.doc_id for p in index.postings.get(term, ())]


def boolean_search(index: InvertedIndex, query: str, engine: Engine = "merge") -> list[int]:
    """Evaluate ``query`` left to right; returns matching doc ids in ascending order."""
    ops = MERGE_OPS if engine == "merge" else SET_OPS
    result: list[int] | None = None
    op = "AND"
    for word in query.split():
        if word in OPERATORS:
            op = word
            continue
        for term in tokenize(word):
            ids = doc_ids(index, term)
            if result is None:  # first operand; a leading NOT means "everything except"
                result = ops["NOT"](sorted(index.doc_meta), ids) if op == "NOT" else ids
            else:
                result = ops[op](result, ids)
            op = "AND"
    return result or []


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m findex.search", description=__doc__)
    parser.add_argument("index", type=Path, help="index file written by findex.index")
    parser.add_argument("query")
    parser.add_argument("--engine", choices=("merge", "set"), default="merge")
    parser.add_argument("--limit", type=int, default=10, help="how many matches to print")
    args = parser.parse_args(argv)

    tracemalloc.start()
    t0 = time.perf_counter()
    index = load(args.index)
    loaded = time.perf_counter()
    matches = boolean_search(index, args.query, args.engine)
    searched = time.perf_counter()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"{len(matches):,} matching documents for {args.query!r} ({args.engine} engine)")
    for doc_id in matches[: args.limit]:
        meta = index.doc_meta[doc_id]
        print(f"  [{doc_id:>5}] {meta.title}  ({meta.path})")
    if len(matches) > args.limit:
        print(f"  ... and {len(matches) - args.limit:,} more")
    print(f"load:        {(loaded - t0) * 1000:.1f} ms")
    print(f"search:      {(searched - loaded) * 1000:.2f} ms")
    print(f"peak memory: {peak / 2**20:.1f} MiB (tracemalloc)")


if __name__ == "__main__":
    main()
