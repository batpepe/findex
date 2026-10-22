"""Where the bytes go: three representations of the same postings (Lab 2, M4).

    uv run python scripts/bench_memory.py data/

Each representation is built in a fresh child process under tracemalloc:

* ``plain``  - ``list[PlainPosting]``, a frozen ``@dataclass`` *without* slots
* ``slots``  - ``list[Posting]``, the project's ``@dataclass(frozen=True, slots=True)``
* ``arrays`` - ``(array('I') doc_ids, array('I') tfs)`` per term, no per-posting objects

and reports the build peak, the memory still held by the finished postings,
the pickled file size and the pickle load time (median of 3).
"""

from __future__ import annotations

import json
import pickle
import statistics
import subprocess
import sys
import tempfile
import time
import tracemalloc
from array import array
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from findex.corpus import iter_documents
from findex.index import Posting
from findex.tokenize import tokenize


@dataclass(frozen=True)
class PlainPosting:
    doc_id: int
    tf: int
    positions: tuple[int, ...] = ()


def build(kind: str, root: Path) -> dict[str, Any]:
    lists: defaultdict[str, list[Any]] = defaultdict(list)
    arrays: defaultdict[str, tuple[array[int], array[int]]] = defaultdict(lambda: (array("I"), array("I")))
    cls = PlainPosting if kind == "plain" else Posting
    for doc in iter_documents(root):
        for term, tf in Counter(tokenize(doc.text)).items():
            if kind == "arrays":
                ids, tfs = arrays[term]
                ids.append(doc.doc_id)
                tfs.append(tf)
            else:
                lists[term].append(cls(doc.doc_id, tf))
    return dict(arrays) if kind == "arrays" else dict(lists)


def child(kind: str, root: Path) -> None:
    tracemalloc.start()
    postings = build(kind, root)
    held, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "postings.pkl"
        with path.open("wb") as f:
            pickle.dump(postings, f, protocol=pickle.HIGHEST_PROTOCOL)
        loads: list[float] = []
        for _ in range(3):
            t0 = time.perf_counter()
            with path.open("rb") as f:
                pickle.load(f)
            loads.append(time.perf_counter() - t0)
        size = path.stat().st_size
    n = sum(len(v) if kind != "arrays" else len(v[0]) for v in postings.values())
    print(json.dumps({"peak": peak, "held": held, "size": size, "load": statistics.median(loads), "n": n}))


def main() -> None:
    if sys.argv[1] == "--child":
        child(sys.argv[2], Path(sys.argv[3]))
        return
    root = sys.argv[1]
    labels = {
        "plain": "`list[Posting]`, plain `@dataclass`",
        "slots": "`list[Posting]`, `slots=True`",
        "arrays": "`array('I')` pairs (doc_ids, tfs)",
    }
    print(
        "| Postings representation | Peak memory (build) | Held by postings | Bytes / posting "
        "| Pickle file size | Pickle load time |"
    )
    print("|---|---|---|---|---|---|")
    for kind, label in labels.items():
        out = subprocess.run(
            [sys.executable, __file__, "--child", kind, root], check=True, capture_output=True, text=True
        ).stdout
        r = json.loads(out)
        mib = 2**20
        print(
            f"| {label} | {r['peak'] / mib:.1f} MiB | {r['held'] / mib:.1f} MiB | {r['held'] / r['n']:.1f} "
            f"| {r['size'] / mib:.1f} MiB | {r['load'] * 1000:.0f} ms |"
        )


if __name__ == "__main__":
    main()
