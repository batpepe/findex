"""Eager vs lazy benchmark for the README (Lab 1, M4).

Runs ``python -m findex.stats`` in a fresh process for every configuration
(so one run's garbage never pollutes the next), 3 times each, and prints a
Markdown table with the median elapsed time and peak memory.

    uv run python scripts/bench_stats.py data/
"""

from __future__ import annotations

import re
import statistics
import subprocess
import sys

RUNS = 3
LIMITS = (400, 800, None)


def run(root: str, limit: int | None, eager: bool) -> dict[str, str]:
    cmd = [sys.executable, "-m", "findex.stats", root, "--top", "0"]
    if limit is not None:
        cmd += ["--limit", str(limit)]
    if eager:
        cmd.append("--eager")
    out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
    return dict(re.findall(r"^([a-zA-Z ]+):\s+(.+)$", out, flags=re.MULTILINE))


def mib(value: str) -> float:
    number, unit = value.split()[:2]
    return float(number) * {"B": 1 / 2**20, "KiB": 1 / 1024, "MiB": 1, "GiB": 1024}[unit]


def main() -> None:
    root = sys.argv[1] if len(sys.argv) > 1 else "data/"
    print("| Version | Documents | Tokens | Peak memory (tracemalloc) | Max RSS | Elapsed (median of 3) |")
    print("|---|---|---|---|---|---|")
    for limit in LIMITS:
        for eager in (True, False):
            runs = [run(root, limit, eager) for _ in range(RUNS)]
            peak = statistics.median(mib(r["peak memory"]) for r in runs)
            rss = statistics.median(mib(r["max RSS"]) for r in runs)
            elapsed = statistics.median(float(r["elapsed"].split()[0]) for r in runs)
            name = "eager (lists)" if eager else "lazy (generators)"
            first = runs[0]
            print(
                f"| {name} | {first['documents']} | {first['tokens']} | {peak:.1f} MiB "
                f"| {rss:.1f} MiB | {elapsed:.2f} s |"
            )


if __name__ == "__main__":
    main()
