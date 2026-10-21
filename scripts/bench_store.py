"""Index file formats: size, save time, load time (Lab 2, M3 and the positions cost).

    uv run python scripts/bench_store.py data/

Builds the index with and without positions, saves it in both formats and
loads each file 5 times in-process (median). For the lazily loaded binary
format, "load + first query" also decodes the postings of a 3-term query, so
the comparison with pickle (which decodes everything up front) stays honest.
"""

from __future__ import annotations

import statistics
import sys
import tempfile
import time
from pathlib import Path

from findex.corpus import iter_documents
from findex.index import build_index
from findex.search import boolean_search
from findex.store import Format, load, save

QUERY = "asyncio event loop"


def timed(fn: object) -> float:
    t0 = time.perf_counter()
    fn()  # type: ignore[operator]
    return time.perf_counter() - t0


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "data")
    print("| Positions | Format | File size | Save | Load (median of 5) | Load + first query |")
    print("|---|---|---|---|---|---|")
    with tempfile.TemporaryDirectory() as tmp:
        for positions in (False, True):
            index = build_index(iter_documents(root), positions=positions)
            fmt: Format
            for fmt in ("pickle", "binary"):
                path = Path(tmp) / f"index-{positions}.{fmt}"
                save_s = timed(lambda: save(index, path, fmt))  # noqa: B023 - called immediately
                loads = [timed(lambda: load(path)) for _ in range(5)]  # noqa: B023
                first = [timed(lambda: boolean_search(load(path), QUERY)) for _ in range(5)]  # noqa: B023
                size = path.stat().st_size / 2**20
                print(
                    f"| {'yes' if positions else 'no'} | {fmt} | {size:.1f} MiB | {save_s:.2f} s "
                    f"| {statistics.median(loads) * 1000:.0f} ms | {statistics.median(first) * 1000:.0f} ms |"
                )


if __name__ == "__main__":
    main()
