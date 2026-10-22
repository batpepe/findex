"""precision@5 of each scorer on the hand-labelled queries in eval/queries.json.

    uv run python scripts/evaluate.py index.bin [eval/queries.json]

precision@5 = |top 5 ∩ relevant| / 5. A query with fewer than 5 relevant
documents cannot reach 1.0; that is fine, both scorers face the same ceiling.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

from findex.index import open_index
from findex.rank import SCORERS, Searcher

K = 5


def main() -> None:
    index_path = Path(sys.argv[1] if len(sys.argv) > 1 else "index.bin")
    labels = json.loads(Path(sys.argv[2] if len(sys.argv) > 2 else "eval/queries.json").read_text())
    names = list(SCORERS)
    totals: dict[str, list[float]] = {name: [] for name in names}
    with open_index(index_path) as index:
        searcher = Searcher(index)
        print("| Query | relevant | " + " | ".join(f"{n} P@5" for n in names) + " |")
        print("|---|---|" + "---|" * len(names))
        for item in labels["queries"]:
            relevant: set[str] = set(item["relevant"])
            row: list[str] = []
            for name in names:
                top = searcher.search(item["query"], SCORERS[name], k=K, snippets=False)
                p = sum(r.path in relevant for r in top) / K
                totals[name].append(p)
                row.append(f"{p:.1f}")
            print(f"| `{item['query']}` | {len(relevant)} | " + " | ".join(row) + " |")
        means = " | ".join(f"**{statistics.mean(totals[n]):.2f}**" for n in names)
        print(f"| **mean** | | {means} |")


if __name__ == "__main__":
    main()
