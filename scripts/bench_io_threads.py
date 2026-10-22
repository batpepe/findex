"""The counter-benchmark: an I/O-bound job where plain (GIL) threads DO scale (Lab 5, bonus).

    uv run python scripts/bench_io_threads.py

Fetches 40 pages of the Python library reference (the corpus' own pages, from
the site whose robots.txt allows it), with 1, 4 and 16 threads, after one
untimed warm-up pass. While a thread waits for the network it has released
the GIL, so another thread runs.
"""

from __future__ import annotations

import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

USER_AGENT = "findex-lab5-io-benchmark/0.4 (+https://github.com/batpepe/findex)"


def fetch(url: str) -> int:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=20) as response:
        return len(response.read())


def main() -> None:
    names = sorted(p.stem for p in Path("data/pydocs/library").glob("*.txt"))[:40]
    urls = [f"https://docs.python.org/3/library/{name}.html" for name in names]
    with ThreadPoolExecutor(16) as pool:  # warm-up, untimed: DNS, TLS session, CDN caches
        list(pool.map(fetch, urls))
    print("| Threads | Pages | Wall (s) | Speedup |")
    print("|---|---|---|---|")
    baseline = None
    for workers in (1, 4, 16):
        t0 = time.perf_counter()
        with ThreadPoolExecutor(workers) as pool:
            total = sum(pool.map(fetch, urls))
        wall = time.perf_counter() - t0
        baseline = baseline or wall
        print(f"| {workers} | {len(urls)} ({total / 2**20:.1f} MiB) | {wall:.2f} | {baseline / wall:.1f}× |")


if __name__ == "__main__":
    main()
