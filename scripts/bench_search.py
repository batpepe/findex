"""End-to-end search latency: parse, match, rank and the 10 snippets, caches off (Lab 8).

    uv run python scripts/bench_search.py index.bin [query ...]

    # the same script against the Lab 7 code, for the before/after table:
    git worktree add ../findex-lab07 lab-07
    PYTHONPATH=../findex-lab07/src uv run python scripts/bench_search.py index.bin

Prints one line per query: the median of 9 searches, each a new ``Searcher``
call with the query cache off (``cache_size=0``) - what a query nobody asked
before costs. Uses only the API that exists since Lab 3, so it runs on both.
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

from findex.rank import BM25, Searcher
from findex.store import load

QUERIES = [
    "python",
    "walrus",
    "global interpreter lock",
    "the",
    '"event loop" NOT asyncio',
    "unicode normalization",
]


def main() -> None:
    index = load(Path(sys.argv[1] if len(sys.argv) > 1 else "index.bin"))
    searcher = Searcher(index, cache_size=0)
    for query in sys.argv[2:] or QUERIES:
        times: list[float] = []
        for _ in range(9):
            t0 = time.perf_counter()
            searcher.search(query, BM25(), 10)
            times.append(time.perf_counter() - t0)
        print(f"| `{query}` | {statistics.median(times) * 1000:.2f} ms |")


if __name__ == "__main__":
    main()
