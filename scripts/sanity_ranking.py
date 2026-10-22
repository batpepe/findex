"""The three ranking sanity checks of Lab 3, on the real corpus.

uv run python scripts/sanity_ranking.py index.bin
"""

from __future__ import annotations

import sys
from pathlib import Path

from findex.index import Index, Posting, open_index
from findex.rank import BM25, Searcher, TfIdf


def rare_beats_common(index: Index) -> None:
    query = "python OR walrus"
    print(f"1) rare term beats common term: {query!r}")
    print(f"   df(python) = {index.df('python')}, df(walrus) = {index.df('walrus')} of {index.num_docs}")
    for result in Searcher(index).search(query, BM25(), k=4, snippets=False):
        has = "walrus" if any(p.doc_id == result.doc_id for p in index["walrus"]) else "python only"
        print(f"   {result.score:7.3f}  [{has}]  {result.title}")


def saturation(index: Index) -> None:
    bm25, tfidf = BM25(), TfIdf()
    doc_id = min(index.doc_lengths, key=lambda d: abs(index.doc_lengths[d] - index.avg_doc_length))
    print(
        f"\n2) term-frequency saturation for 'walrus' in an average-length document "
        f"({index.doc_lengths[doc_id]} tokens, avg {index.avg_doc_length:.0f})"
    )
    print("   tf | BM25   | gain vs tf-1 | TF-IDF | gain vs tf-1")
    previous = None
    for tf in (1, 2, 3, 5, 10, 19, 20, 21):
        b = bm25.score("walrus", Posting(doc_id, tf), index)
        t = tfidf.score("walrus", Posting(doc_id, tf), index)
        if previous is not None and tf - 1 == previous[0]:
            print(f"   {tf:>2} | {b:6.3f} | {b - previous[1]:+.4f}      | {t:6.3f} | {t - previous[2]:+.4f}")
        else:
            print(f"   {tf:>2} | {b:6.3f} |              | {t:6.3f} |")
        previous = (tf, b, t)


def short_beats_long(index: Index) -> None:
    # a term that occurs exactly once in both a very short and a very long document
    for term in sorted(index, key=index.df):
        once = [p for p in index[term] if p.tf == 1]
        if len(once) < 2 or index.df(term) > 20 or not term.isalpha():
            continue
        short = min(once, key=lambda p: index.doc_length(p.doc_id))
        long = max(once, key=lambda p: index.doc_length(p.doc_id))
        if index.doc_length(short.doc_id) < 300 and index.doc_length(long.doc_id) > 50_000:
            break
    else:  # pragma: no cover
        print("no suitable term")
        return
    print(f"\n3) one match in a short vs a long document: term {term!r} (df={index.df(term)})")
    for posting in (short, long):
        bm25, tfidf = BM25().score(term, posting, index), TfIdf().score(term, posting, index)
        print(
            f"   {index.doc_length(posting.doc_id):>7,} tokens  BM25 {bm25:6.3f}"
            f"  TF-IDF {tfidf:6.3f}  {index.title(posting.doc_id)}"
        )


def main() -> None:
    with open_index(Path(sys.argv[1] if len(sys.argv) > 1 else "index.bin")) as index:
        rare_beats_common(index)
        saturation(index)
        short_beats_long(index)


if __name__ == "__main__":
    main()
