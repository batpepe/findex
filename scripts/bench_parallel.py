# pyright: reportUnknownMemberType=false
# (matplotlib's type annotations are incomplete; everything else here is strictly typed)
"""Serial vs threads vs processes (vs free-threaded threads) for building the index (Lab 5, M3/M4).

    uv run python scripts/bench_parallel.py data/ [--ft .venv-ft/bin/python] [--runs 3]

Every configuration runs in a fresh child process (so peak RSS is that run's
alone), ``--runs`` times; the table reports medians. CPU time is user + system
of the child *and* its worker processes; wall time is what you wait for.
Writes docs/bench-parallel.json and docs/speedup.png.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import sysconfig
import time
from pathlib import Path
from typing import Any

WORKERS = (1, 2, 4, 8, os.cpu_count() or 8)


def child(root: Path, executor: str, workers: int) -> None:
    from findex.parallel import BuildReport, ExecutorName, build_parallel

    name: ExecutorName = executor  # type: ignore[assignment]  # validated by argparse choices
    report = BuildReport()
    t0 = time.perf_counter()
    build_parallel(root, executor=name, workers=workers, report=report)
    wall = time.perf_counter() - t0
    me, kids = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
    rss = me.ru_maxrss * (1 if sys.platform == "darwin" else 1024)
    gil = getattr(sys, "_is_gil_enabled", lambda: True)()
    print(
        json.dumps(
            {
                "wall": wall,
                "cpu": me.ru_utime + me.ru_stime + kids.ru_utime + kids.ru_stime,
                "rss": rss,
                "map": report.map_seconds,
                "merge": report.merge_seconds,
                "chunks": report.chunks,
                "gil": gil,
            }
        )
    )


def run(python: str, root: str, executor: str, workers: int) -> dict[str, Any]:
    cmd = [python, __file__, "--child", executor, str(workers), root]
    out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
    return json.loads(out.splitlines()[-1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", default="data")
    parser.add_argument("--child", nargs=2, metavar=("EXECUTOR", "WORKERS"))
    parser.add_argument("--ft", help="path to a free-threaded python with findex installed")
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()
    if args.child:
        child(Path(args.root), args.child[0], int(args.child[1]))
        return

    configs: list[tuple[str, str, str, int]] = [("serial", sys.executable, "serial", 1)]
    configs += [("threads", sys.executable, "threads", w) for w in WORKERS]
    configs += [("processes", sys.executable, "processes", w) for w in WORKERS]
    if args.ft:
        configs += [("threads (3.14t, no GIL)", args.ft, "threads", w) for w in WORKERS]

    rows: list[dict[str, Any]] = []
    baseline = None
    for label, python, executor, workers in configs:
        runs = [run(python, args.root, executor, workers) for _ in range(args.runs)]
        row: dict[str, Any] = {"label": label, "workers": workers, "gil": runs[0]["gil"]}
        for key in ("wall", "cpu", "rss", "map", "merge"):
            row[key] = statistics.median(r[key] for r in runs)
        row["wall_min"], row["wall_max"] = min(r["wall"] for r in runs), max(r["wall"] for r in runs)
        row["chunks"] = runs[0]["chunks"]
        baseline = baseline or row["wall"]
        row["speedup"] = baseline / row["wall"]
        rows.append(row)
        print(
            f"| {label} | {workers} | {row['wall']:.2f} ({row['wall_min']:.2f}-{row['wall_max']:.2f}) "
            f"| {row['cpu']:.2f} | {row['rss'] / 2**20:.0f} MiB | {row['merge']:.2f} "
            f"| {row['speedup']:.2f}× |",
            flush=True,
        )

    docs = Path(__file__).parents[1] / "docs"
    docs.mkdir(exist_ok=True)
    meta = {
        "machine": f"{platform.machine()} {platform.processor() or ''}".strip(),
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "free_threaded_build": bool(sysconfig.get_config_var("Py_GIL_DISABLED")),
        "runs": args.runs,
    }
    (docs / "bench-parallel.json").write_text(json.dumps({"meta": meta, "rows": rows}, indent=2))
    plot(rows, docs / "speedup.png")


def plot(rows: list[dict[str, Any]], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    workers = sorted({r["workers"] for r in rows if r["label"] != "serial"})
    ax.plot(workers, workers, linestyle="--", color="#9ca3af", label="ideal (linear)")
    styles = {"threads": "#d97706", "processes": "#2563eb", "threads (3.14t, no GIL)": "#059669"}
    for label, color in styles.items():
        series = [r for r in rows if r["label"] == label]
        if series:
            ax.plot(
                [r["workers"] for r in series],
                [r["speedup"] for r in series],
                marker="o",
                color=color,
                label=label,
            )
    ax.axhline(1, color="#6b7280", linewidth=0.8)
    ax.set_xlabel("workers")
    ax.set_ylabel("speedup vs serial (wall time)")
    ax.set_title("findex index: speedup vs workers (Apple M4, 4P+6E cores)")
    ax.set_xticks(workers)
    ax.set_ylim(0, max(workers) + 0.5)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out)


if __name__ == "__main__":
    main()
