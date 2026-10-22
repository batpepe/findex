"""`def` vs `async def` for /search, and the oha load-test table (Lab 7, M2 + M4).

    uv run python scripts/bench_web.py index.bin [.venv-ft/bin/python]   # needs `oha` on PATH

Each configuration gets a fresh uvicorn on its own port with the query cache
off (SEARCH_CACHE_SIZE=0), so every request really ranks. Two experiments:

1. 20 concurrent searches, plus one /health probe sent 20 ms later: total time,
   search latencies, and how long the *probe* waited - i.e. whether the event
   loop stayed free while the searches ran. Median of 5 rounds.
2. oha, 15 s at concurrency 50, cycling through 20 realistic queries, no keep-alive;
   median of 3 runs.
"""

from __future__ import annotations

import asyncio
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.parse
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

import httpx

QUERIES = [
    "asyncio event loop", "global interpreter lock", "structural pattern matching", "type hints generics",
    "virtual environment", "f-string formatting", "async generator", "dataclass frozen slots",
    "context manager", "unicode normalization", "reference counting", "build backend pyproject",
    '"event loop" NOT asyncio', "pip install requirements", "garbage collector", "free-threaded",
    "import system", "decorator functools wraps", "subprocess pipe", "logging handler",
]  # fmt: skip


@contextmanager
def server(
    index: Path, port: int, *, workers: int = 1, search_async: bool = False, python: str = sys.executable
) -> Generator[str]:
    env = os.environ | {
        "INDEX_PATH": str(index.resolve()),
        "SEARCH_CACHE_SIZE": "0",
        "SEARCH_ASYNC": "true" if search_async else "false",
        "LOG_LEVEL": "warning",
    }
    cmd = [
        python,
        "-m",
        "uvicorn",
        "findex.web.app:app",
        "--port",
        str(port),
        "--workers",
        str(workers),
    ]
    proc = subprocess.Popen([*cmd, "--log-level", "warning"], env=env)
    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(100):
            try:
                if httpx.get(f"{base}/health", timeout=1).status_code == 200:
                    break
            except httpx.TransportError:
                pass
            time.sleep(0.1)
        yield base
    finally:
        proc.terminate()
        proc.wait(timeout=10)


async def burst(base: str) -> tuple[float, float, float]:
    async with httpx.AsyncClient(base_url=base, timeout=60) as client:
        t0 = time.perf_counter()

        async def one(q: str) -> float:
            s = time.perf_counter()
            (await client.get("/search", params={"q": q, "k": 10})).raise_for_status()
            return time.perf_counter() - s

        async def probe() -> float:
            await asyncio.sleep(0.02)
            s = time.perf_counter()
            await client.get("/health")
            return time.perf_counter() - s

        *latencies, health = await asyncio.gather(*(one(q) for q in QUERIES), probe())
        return time.perf_counter() - t0, statistics.median(latencies), health


def oha(base: str, seconds: int = 15, concurrency: int = 50) -> dict[str, float]:
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        for q in QUERIES:
            f.write(f"{base}/search?q={urllib.parse.quote_plus(q)}&k=10\n")
    # --disable-keepalive: a new connection per request, so with several uvicorn processes an idle
    # one accepts it; with 50 long-lived connections, one busy process can end up holding most of them
    cmd = ["oha", "--no-tui", "--output-format", "json", "-z", f"{seconds}s", "-c", str(concurrency),
           "--disable-keepalive", "--urls-from-file", f.name]  # fmt: skip
    report = json.loads(subprocess.run(cmd, check=True, capture_output=True, text=True).stdout)
    Path(f.name).unlink()
    pct = report["latencyPercentiles"]
    return {
        "rps": report["summary"]["requestsPerSec"],
        "p50": pct["p50"] * 1000,
        "p95": pct["p95"] * 1000,
        "p99": pct["p99"] * 1000,
        "ok": report["summary"]["successRate"],
    }


def main() -> None:
    index = Path(sys.argv[1] if len(sys.argv) > 1 else "index.bin")
    ft = sys.argv[2] if len(sys.argv) > 2 else None  # a free-threaded python with findex installed
    print("20 concurrent searches + a /health probe (median of 5 rounds):\n")
    print("| /search is | All 20 done (s) | Median search (ms) | /health probe waited (ms) |")
    print("|---|---|---|---|")
    variants = [
        ("`async def`", 8101, True, sys.executable),
        ("`def` (thread pool)", 8102, False, sys.executable),
    ]
    if ft:
        variants.append(("`def` (thread pool), 3.14t no GIL", 8106, False, ft))
    for label, port, search_async, python in variants:
        with server(index, port, search_async=search_async, python=python) as base:
            asyncio.run(burst(base))  # warm-up
            rounds = [asyncio.run(burst(base)) for _ in range(5)]
        total, median, health = (statistics.median(r[i] for r in rounds) for i in range(3))
        print(f"| {label} | {total:.2f} | {median * 1000:.0f} | {health * 1000:.0f} |")

    print("\noha, 15 s at concurrency 50, 20 rotating queries, query cache off, median of 3:\n")
    print("| Target | Route | Concurrency | RPS | p50 (ms) | p95 | p99 |")
    print("|---|---|---|---|---|---|---|")
    configs = [
        ("local, 1 worker, `async def`", 8103, 1, True, sys.executable),
        ("local, 1 worker, `def`", 8104, 1, False, sys.executable),
        ("local, 4 workers, `def`", 8105, 4, False, sys.executable),
    ]
    if ft:
        configs.append(("local, 1 worker, `def`, 3.14t no GIL", 8107, 1, False, ft))
    for label, port, workers, search_async, python in configs:
        with server(index, port, workers=workers, search_async=search_async, python=python) as base:
            oha(base, seconds=3)  # warm-up: page cache, imports, first-touch of the index
            runs = [oha(base) for _ in range(3)]
        r = {key: statistics.median(run[key] for run in runs) for key in runs[0]}
        latency = f"{r['p50']:.0f} | {r['p95']:.0f} | {r['p99']:.0f}"
        print(f"| {label} | `/search` | 50 | {r['rps']:.0f} | {latency} |")


if __name__ == "__main__":
    main()
