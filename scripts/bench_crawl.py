"""Crawler throughput at concurrency 1 / 5 / 20 (Lab 6, M4).

    uv run python scripts/bench_crawl.py real     # 4 Read the Docs sites: 2 per host, 1.2 s apart
    uv run python scripts/bench_crawl.py mirror   # local mirror (scripts/mirror_server.py must be running)

Every configuration runs in a fresh process, so peak RSS is that run's own.
"""

from __future__ import annotations

import asyncio
import json
import resource
import subprocess
import sys
import time
from typing import Any

from findex.crawler import CrawlStats, crawl

PROFILES: dict[str, dict[str, Any]] = {
    "real": {
        "seeds": [
            "https://packaging.python.org/en/latest/",
            "https://devguide.python.org/",
            "https://typing.python.org/en/latest/",
            "https://pip.pypa.io/en/stable/",
        ],
        "max_pages": 200,
        "per_host": 2,
        "delay": 1.2,  # 4 hosts x 1/1.2 s = 3.3 requests/s in total: under Read the Docs' limit of 4/s
    },
    "mirror": {
        "seeds": ["http://127.0.0.1:8765/"],
        "max_pages": 300,
        "per_host": 1000,  # my own server: no politeness limit, so concurrency is all that counts
        "delay": 0.0,
    },
}


def child(profile: str, concurrency: int) -> None:
    settings = PROFILES[profile]
    stats = CrawlStats()

    async def go() -> None:
        pages = crawl(
            settings["seeds"],
            max_pages=settings["max_pages"],
            concurrency=concurrency,
            per_host=settings["per_host"],
            delay=settings["delay"],
            stats=stats,
        )
        async for _ in pages:
            pass

    t0 = time.perf_counter()
    asyncio.run(go())
    wall = time.perf_counter() - t0
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
    print(
        json.dumps(
            {"pages": stats.pages, "fetched": stats.fetched, "errors": stats.errors, "wall": wall, "rss": rss}
        )
    )


def main() -> None:
    if sys.argv[1] == "--child":
        child(sys.argv[2], int(sys.argv[3]))
        return
    profile = sys.argv[1]
    print("| Concurrency | Pages | Fetched | Wall (s) | Pages/s | Errors | Peak RSS |")
    print("|---|---|---|---|---|---|---|")
    for concurrency in (1, 5, 20):
        cmd = [sys.executable, __file__, "--child", profile, str(concurrency)]
        r = json.loads(
            subprocess.run(cmd, check=True, capture_output=True, text=True).stdout.splitlines()[-1]
        )
        rate = r["pages"] / r["wall"]
        print(
            f"| {concurrency} | {r['pages']} | {r['fetched']} | {r['wall']:.1f} | {rate:.1f} "
            f"| {r['errors']} | {r['rss'] / 2**20:.0f} MiB |",
            flush=True,
        )


if __name__ == "__main__":
    main()
