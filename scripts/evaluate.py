"""precision@5 on the hand-labelled queries in eval/queries.json.

    uv run python scripts/evaluate.py index.bin            # Lab 3: BM25 vs TF-IDF, the 10 original queries
    uv run python scripts/evaluate.py index.bin --modes    # Lab 8: keyword / semantic / hybrid, all 15
                                                           #   (needs `findex embed index.bin` first)

precision@5 = |top 5 ∩ relevant| / 5. A query with fewer than 5 relevant
documents cannot reach 1.0; that is fine, every column faces the same ceiling.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Callable
from pathlib import Path
from typing import Any

from findex.index import open_index
from findex.rank import MODES, SCORERS, Searcher
from findex.semantic import get_embedder, load_vectors

K = 5


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("index", type=Path, nargs="?", default=Path("index.bin"))
    parser.add_argument("labels", type=Path, nargs="?", default=Path("eval/queries.json"))
    parser.add_argument("--modes", action="store_true", help="compare keyword / semantic / hybrid (BM25)")
    args = parser.parse_args()
    items = json.loads(args.labels.read_text())["queries"]

    with open_index(args.index) as index:
        if args.modes:
            vectors = load_vectors(args.index)
            if vectors is None:
                raise SystemExit(f"no embeddings next to {args.index}: run `findex embed {args.index}` first")
            searcher = Searcher(index, vectors=vectors, embedder=get_embedder(vectors.model))
            columns: dict[str, Callable[[str], list[str]]] = {
                mode: lambda q, m=mode: [r.path for r in searcher.search(q, k=K, snippets=False, mode=m)]
                for mode in MODES
            }
            print(f"embeddings: {vectors.model}, {len(vectors):,} chunks")
        else:
            searcher = Searcher(index)
            items = [item for item in items if not item.get("paraphrase")]
            columns = {
                name: lambda q, s=scorer: [r.path for r in searcher.search(q, s, k=K, snippets=False)]
                for name, scorer in SCORERS.items()
            }
        table(items, columns)


def table(items: list[dict[str, Any]], columns: dict[str, Callable[[str], list[str]]]) -> None:
    totals: dict[str, list[float]] = {name: [] for name in columns}
    paraphrased: dict[str, list[float]] = {name: [] for name in columns}
    print("| Query | relevant | " + " | ".join(f"{name} P@5" for name in columns) + " |")
    print("|---|---|" + "---|" * len(columns))
    for item in items:
        query = str(item["query"])
        relevant: set[str] = set(item["relevant"])
        row: list[str] = []
        for name, top in columns.items():
            p = sum(path in relevant for path in top(query)) / K
            totals[name].append(p)
            if item.get("paraphrase"):
                paraphrased[name].append(p)
            row.append(f"{p:.1f}")
        label = f"*{query}* (paraphrase)" if item.get("paraphrase") else f"`{query}`"
        print(f"| {label} | {len(relevant)} | " + " | ".join(row) + " |")

    def means(label: str, values: dict[str, list[float]], bold: bool = False) -> None:
        cells = (f"{statistics.mean(values[name]):.2f}" for name in columns)
        print(f"| {label} | | " + " | ".join(f"**{c}**" if bold else c for c in cells) + " |")

    if any(paraphrased.values()):
        original = {name: totals[name][: len(totals[name]) - len(paraphrased[name])] for name in columns}
        means("mean, original 10", original)
        means("mean, 5 paraphrases", paraphrased)
    means("**mean**", totals, bold=True)


if __name__ == "__main__":
    main()
