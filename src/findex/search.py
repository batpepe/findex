"""Search the index from the command line.

    python -m findex.search index.bin "query" [--scorer bm25|tfidf] [-k 10] [--explain] [-v]
    python -m findex.search index.bin "query" --boolean [--engine merge|set]

Ranked search (default, Lab 3) understands the full query language of
``findex.query``: implicit AND, ``OR``, ``NOT``, parentheses and
``"quoted phrases"``. ``--boolean`` keeps the Lab 2 evaluator: words ANDed,
``OR``/``NOT`` strictly left to right, unranked, with the two-pointer merge or
Python sets (``--engine``)::

    asyncio task        ->  asyncio AND task
    thread OR process   ->  thread OR process
    python NOT java     ->  python AND NOT java
    a OR b c            ->  (a OR b) AND c      (Lab 2: no precedence)
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Literal

from findex.index import Index
from findex.merge import MERGE_OPS, SET_OPS
from findex.store import load
from findex.tokenize import tokenize

type Engine = Literal["merge", "set"]

OPERATORS = frozenset({"AND", "OR", "NOT"})


def doc_ids(index: Index, term: str) -> list[int]:
    return [p.doc_id for p in index.get(term, ())]


def boolean_search(index: Index, query: str, engine: Engine = "merge") -> list[int]:
    """Lab 2: evaluate ``query`` left to right; matching doc ids in ascending order."""
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


def _print_boolean(index: Index, query: str, engine: Engine, limit: int) -> None:
    matches = boolean_search(index, query, engine)
    print(f"{len(matches):,} matching documents for {query!r} ({engine} engine)")
    for doc_id in matches[:limit]:
        meta = index.doc_meta[doc_id]
        print(f"  [{doc_id:>5}] {meta.title}  ({meta.path})")
    if len(matches) > limit:
        print(f"  ... and {len(matches) - limit:,} more")


def _print_ranked(index: Index, query: str, scorer_name: str, k: int, repeat: int, explain: bool) -> None:
    from findex.query import explain as explain_tree
    from findex.query import parse
    from findex.rank import SCORERS, Searcher, explain_score

    searcher = Searcher(index)
    scorer = SCORERS[scorer_name]
    results = searcher.search(query, scorer, k)
    for _ in range(repeat - 1):  # repeated query: the LRU cache answers (see -v timings)
        results = searcher.search(query, scorer, k)
    print(f"{results.total:,} matching documents for {query!r} ({scorer_name}, top {len(results)})")
    if explain:
        node = parse(query)
        print(f"parsed as: {explain_tree(node)}")
    for rank, result in enumerate(results, start=1):
        print(f"{rank:>3}. {result}")
        if explain:
            for term, tf, df, contribution in explain_score(index, parse(query), result.doc_id, scorer):
                print(f"           {term:<16} tf={tf:<5} df={df:<5} -> {contribution:6.3f}")
            print(
                f"           doc length {index.doc_length(result.doc_id):,} tokens "
                f"(average {index.avg_doc_length:,.0f})"
            )


def main(argv: list[str] | None = None) -> None:
    from findex.query import QueryError

    parser = argparse.ArgumentParser(prog="python -m findex.search", description=__doc__)
    parser.add_argument("index", type=Path, help="index file written by findex.index")
    parser.add_argument("query")
    parser.add_argument("--scorer", choices=("bm25", "tfidf"), default="bm25")
    parser.add_argument("-k", type=int, default=10, help="how many results to show")
    parser.add_argument("--boolean", action="store_true", help="Lab 2 unranked left-to-right search")
    parser.add_argument("--engine", choices=("merge", "set"), default="merge", help="with --boolean")
    parser.add_argument("--explain", action="store_true", help="show the parse tree and score terms")
    parser.add_argument("--repeat", type=int, default=1, help="run the query N times (cache demo)")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging (timings)")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s"
    )

    tracemalloc.start()
    t0 = time.perf_counter()
    index = load(args.index)
    loaded = time.perf_counter()
    try:
        if args.boolean:
            _print_boolean(index, args.query, args.engine, args.k)
        else:
            _print_ranked(index, args.query, args.scorer, args.k, args.repeat, args.explain)
    except QueryError as exc:
        print(f"error: bad query: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
    searched = time.perf_counter()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"load:        {(loaded - t0) * 1000:.1f} ms")
    print(f"search:      {(searched - loaded) * 1000:.2f} ms")
    print(f"peak memory: {peak / 2**20:.1f} MiB (tracemalloc)")


if __name__ == "__main__":
    main()
