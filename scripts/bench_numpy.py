"""Lab 3's Python scorer vs the NumPy engine on the same index (Lab 8, M2).

    uv run python scripts/bench_numpy.py index.bin [query ...]

Ranking only - parse, Boolean match, score, top 10 - with the query cache off
and no snippets, so both engines do exactly the same job. Each cell is the
median of 7 repeats; a repeat runs the query enough times to last ~0.2 s.
The two engines' rankings are checked to be identical before timing.
"""

from __future__ import annotations

import statistics
import sys
import timeit
from pathlib import Path

from findex.query import parse
from findex.rank import BM25, Searcher
from findex.store import load
from findex.vector import evaluate_mask

DEFAULT_QUERIES = [
    ("common term", "python"),
    ("rare term", "walrus"),
    ("3-term query", "global interpreter lock"),
    ("most common term", "the"),
    ("phrase of common words", '"in the"'),
]


def median_ms(fn: object, target: float = 0.2) -> float:
    timer = timeit.Timer(fn)  # pyright: ignore[reportArgumentType]
    number, _ = timer.autorange()
    number = max(1, int(number * target / 0.2))
    return statistics.median(timer.repeat(repeat=7, number=number)) / number * 1000


def main() -> None:
    index = load(Path(sys.argv[1] if len(sys.argv) > 1 else "index.bin"))
    queries = [(q, q) for q in sys.argv[2:]] or DEFAULT_QUERIES
    python = Searcher(index, cache_size=0, engine="python")
    numpy = Searcher(index, cache_size=0, engine="numpy")
    bm25 = BM25()
    print("| Query | Docs matched | Lab 3 scorer (ms) | NumPy scorer (ms) | Speedup |")
    print("|---|---|---|---|---|")
    for label, query in queries:
        expected = python.search(query, bm25, 10, snippets=False)
        got = numpy.search(query, bm25, 10, snippets=False)
        assert [(r.doc_id, r.score) for r in got] == [(r.doc_id, r.score) for r in expected], query
        matched = int(evaluate_mask(parse(query), index).sum())

        def run_python(q: str = query) -> None:
            python.search(q, bm25, 10, snippets=False)

        def run_numpy(q: str = query) -> None:
            numpy.search(q, bm25, 10, snippets=False)

        slow, fast = median_ms(run_python), median_ms(run_numpy)
        name = label if label == query else f"{label} (`{query}`)"
        print(f"| {name} | {matched:,} | {slow:.3f} | {fast:.3f} | {slow / fast:.1f}× |")


if __name__ == "__main__":
    main()
