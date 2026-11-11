"""Run searches in a loop so a profiler sees the steady state (Lab 8, M1).

    uv run python scripts/profile_search.py index.bin "global interpreter lock" [more queries...]
        [--seconds 8] [--engine python|numpy] [--no-snippets]

    # which functions (deterministic, inflated by the instrumentation - trust the ranking):
    uv run python -m cProfile -s tottime scripts/profile_search.py index.bin "global interpreter lock"
    # where the time really goes (sampling, ~no overhead), as a flame graph:
    py-spy record -o docs/flame-search.svg -- .venv/bin/python scripts/profile_search.py index.bin ...
    # Python time vs native time, per line:
    scalene run --cpu-only --profile-all scripts/profile_search.py --- index.bin ...

The query cache is off (``cache_size=0``), so every iteration does what a new
query does: parse, evaluate, score, top-k and snippets.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from findex.rank import ENGINES, Searcher
from findex.store import load


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("index", type=Path)
    parser.add_argument("queries", nargs="+")
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--engine", choices=ENGINES, default="numpy")
    parser.add_argument("--no-snippets", dest="snippets", action="store_false")
    args = parser.parse_args()

    index = load(args.index)
    searcher = Searcher(index, cache_size=0, engine=args.engine)
    count = 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < args.seconds:
        for query in args.queries:
            searcher.search(query, k=10, snippets=args.snippets)
            count += 1
    elapsed = time.perf_counter() - t0
    print(f"{count} searches in {elapsed:.2f} s: {elapsed / count * 1000:.2f} ms each", file=sys.stderr)


if __name__ == "__main__":
    main()
