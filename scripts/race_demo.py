"""What the GIL was hiding: threads sharing one Counter (Lab 5, M4).

    uv run python scripts/race_demo.py data/                 # normal build (GIL)
    .venv-ft/bin/python scripts/race_demo.py data/           # free-threaded build

Four threads count tokens of a quarter of the corpus each, three ways:

* ``shared +=``     - ``counts[token] += 1`` on one shared Counter: read, add, write back;
* ``shared update`` - ``counts.update(tokens)`` on one shared Counter;
* ``per-thread``    - each thread fills its own Counter, merged after ``join`` (no sharing);
* ``locked +=``     - the first variant with a ``threading.Lock`` around each update.

A correct run counts exactly the tokens the serial pipeline counts.
"""

from __future__ import annotations

import sys
import threading
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path

from findex.corpus import iter_documents
from findex.tokenize import tokenize

THREADS = 4


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "data")
    token_lists = [list(tokenize(doc.text)) for doc in iter_documents(root)]  # tokenize once, up front
    expected = sum(map(len, token_lists))
    slices = [token_lists[i::THREADS] for i in range(THREADS)]
    gil = "enabled" if getattr(sys, "_is_gil_enabled", lambda: True)() else "DISABLED"
    print(f"Python {sys.version.split()[0]}, GIL {gil}; expected {expected:,} tokens")

    def run(name: str, work: Callable[[list[list[str]]], None], result: Callable[[], Counter[str]]) -> None:
        threads = [threading.Thread(target=work, args=(part,)) for part in slices]
        t0 = time.perf_counter()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        total = result().total()
        lost = expected - total
        verdict = "ok" if lost == 0 else f"LOST {lost:,} updates"
        print(f"  {name:<14} {total:>10,}  {time.perf_counter() - t0:5.2f} s  {verdict}")

    shared: Counter[str] = Counter()

    def shared_increment(part: list[list[str]]) -> None:
        for tokens in part:
            for token in tokens:
                shared[token] += 1  # read-modify-write: two threads can read the same old value

    run("shared +=", shared_increment, lambda: shared)

    shared_update: Counter[str] = Counter()

    def shared_bulk(part: list[list[str]]) -> None:
        for tokens in part:
            shared_update.update(tokens)

    run("shared update", shared_bulk, lambda: shared_update)

    partials: list[Counter[str]] = []

    def own_counter(part: list[list[str]]) -> None:
        mine: Counter[str] = Counter()
        for tokens in part:
            mine.update(tokens)
        partials.append(mine)  # list.append of a finished object: one atomic operation

    run("per-thread", own_counter, lambda: sum(partials, Counter[str]()))

    locked: Counter[str] = Counter()
    lock = threading.Lock()

    def locked_increment(part: list[list[str]]) -> None:
        for tokens in part:
            for token in tokens:
                with lock:
                    locked[token] += 1

    run("locked +=", locked_increment, lambda: locked)


if __name__ == "__main__":
    main()
