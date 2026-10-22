"""Where the process pool's time goes (Lab 5, the explanation's numbers).

uv run python scripts/analyze_parallel.py data/
"""

from __future__ import annotations

import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from findex.corpus import iter_paths
from findex.parallel import PartialIndex, build_partial, chunk_paths


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "data")
    paths = list(iter_paths(root))

    t0 = time.perf_counter()
    with ProcessPoolExecutor(8) as pool:
        empty: list[list[Path]] = [[] for _ in range(8)]
        list(pool.map(build_partial, empty, [root] * 8))  # empty chunks: start-up + import only
    print(f"spawn 8 workers + import findex + one round trip: {time.perf_counter() - t0:.2f} s")

    chunks = chunk_paths(paths, 32)  # what 8 workers x 4 chunks each get
    sizes = sorted((sum(p.stat().st_size for p in c), len(c)) for c in chunks)
    print(
        f"32 chunks: smallest {sizes[0][0] / 1e6:.2f} MB ({sizes[0][1]} files), "
        f"largest {sizes[-1][0] / 1e6:.2f} MB ({sizes[-1][1]} files)"
    )

    partials: list[PartialIndex] = []
    slowest = 0.0
    for chunk in chunks:
        t = time.perf_counter()
        partials.append(build_partial(chunk, root))
        slowest = max(slowest, time.perf_counter() - t)
    print(f"slowest single chunk, built alone: {slowest:.2f} s (the critical path of the map phase)")

    t = time.perf_counter()
    blobs = [pickle.dumps(p, protocol=pickle.HIGHEST_PROTOCOL) for p in partials]
    dumped = time.perf_counter() - t
    t = time.perf_counter()
    for blob in blobs:
        pickle.loads(blob)
    loaded = time.perf_counter() - t
    total = sum(map(len, blobs))
    print(
        f"pickling the 32 partials: {total / 2**20:.1f} MiB, dumps {dumped:.2f} s (in workers), "
        f"loads {loaded:.2f} s (in the parent, sequential)"
    )


if __name__ == "__main__":
    main()
