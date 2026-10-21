"""Two-pointer merge vs Python sets for Boolean operators (Lab 2, M2).

    uv run python scripts/bench_merge.py index.bin

Picks the two most common and the two rarest terms of the index and times
AND / OR / NOT with both engines (best of 5 repeats of N calls).
"""

from __future__ import annotations

import sys
import timeit
from pathlib import Path

from findex.search import MERGE_OPS, SET_OPS, doc_ids
from findex.store import load


def best_us(fn: object, a: list[int], b: list[int]) -> float:
    number = 2000
    times = timeit.repeat(lambda: fn(a, b), number=number, repeat=5)  # type: ignore[operator]
    return min(times) / number * 1e6


def main() -> None:
    index = load(Path(sys.argv[1] if len(sys.argv) > 1 else "index.bin"))
    by_df = sorted(index.postings, key=lambda t: (len(index.postings[t]), t))
    rare = [t for t in by_df if t.isalpha()][:2]
    common = by_df[-2:][::-1]
    pairs = {
        f"common AND common (`{common[0]}`, `{common[1]}`)": (common[0], common[1]),
        f"rare AND rare (`{rare[0]}`, `{rare[1]}`)": (rare[0], rare[1]),
        f"rare AND common (`{rare[0]}`, `{common[0]}`)": (rare[0], common[0]),
    }
    print("| Operands | len(a), len(b) | Op | merge (µs) | set (µs) | faster |")
    print("|---|---|---|---|---|---|")
    for label, (x, y) in pairs.items():
        a, b = doc_ids(index, x), doc_ids(index, y)
        for op in ("AND", "OR", "NOT"):
            merge, sets = best_us(MERGE_OPS[op], a, b), best_us(SET_OPS[op], a, b)
            winner = f"merge {sets / merge:.1f}×" if merge < sets else f"set {merge / sets:.1f}×"
            print(f"| {label} | {len(a)}, {len(b)} | {op} | {merge:.2f} | {sets:.2f} | {winner} |")


if __name__ == "__main__":
    main()
