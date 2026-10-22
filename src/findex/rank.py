"""Ranking: TF-IDF and BM25 behind one ``Scorer`` protocol, top-k with ``heapq``.

``Searcher.search`` parses the query (``findex.query``), evaluates it to the set
of matching documents, scores those documents term by term with the chosen
scorer, keeps the ``k`` best with ``heapq.nlargest`` (O(n log k), not a full
sort), and attaches a highlighted snippet to each. The query -> ranked ids step
is memoized with ``functools.lru_cache``; repeated queries skip straight to
the snippets.
"""

from __future__ import annotations

import functools
import heapq
import logging
import math
from collections import defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from operator import itemgetter
from typing import Protocol, overload

from findex.index import Index, Posting
from findex.query import Node, parse, positive_terms
from findex.snippets import Snippet, make_snippet
from findex.timing import timed

log = logging.getLogger(__name__)


class Scorer(Protocol):
    """Anything with this method is a scorer - no inheritance needed (structural typing)."""

    def score(self, term: str, posting: Posting, index: Index) -> float: ...


@dataclass(frozen=True, slots=True)
class TfIdf:
    """``(1 + ln tf) * ln(N / df)``: sublinear term frequency, rare terms weigh more.

    No document-length normalization: one match in a 2 MB changelog counts as
    much as one match in a 2 KB page. That is what BM25 fixes.
    """

    name = "tfidf"

    def score(self, term: str, posting: Posting, index: Index) -> float:
        return (1 + math.log(posting.tf)) * math.log(index.num_docs / index.df(term))

    def __call__(self, term: str, posting: Posting, index: Index) -> float:
        return self.score(term, posting, index)


@dataclass(frozen=True, slots=True)
class BM25:
    """Okapi BM25 with Lucene's non-negative IDF.

    ``k1`` controls term-frequency saturation (the 20th occurrence adds almost
    nothing); ``b`` controls document-length normalization (``b = 0``: none,
    ``b = 1``: full).
    """

    k1: float = 1.5
    b: float = 0.75
    name = "bm25"

    def idf(self, term: str, index: Index) -> float:
        df = index.df(term)
        return math.log(1 + (index.num_docs - df + 0.5) / (df + 0.5))

    def score(self, term: str, posting: Posting, index: Index) -> float:
        length_norm = 1 - self.b + self.b * index.doc_length(posting.doc_id) / index.avg_doc_length
        tf = posting.tf
        return self.idf(term, index) * tf * (self.k1 + 1) / (tf + self.k1 * length_norm)

    def __call__(self, term: str, posting: Posting, index: Index) -> float:
        return self.score(term, posting, index)


SCORERS: dict[str, Scorer] = {"bm25": BM25(), "tfidf": TfIdf()}


@dataclass(frozen=True, order=True, slots=True)
class SearchResult:
    # order=True compares fields as a tuple in declaration order: score first,
    # so sorted(results, reverse=True) ranks them and doc_id breaks ties.
    score: float
    doc_id: int
    title: str = field(compare=False)
    path: str = field(compare=False)
    snippet: Snippet | None = field(default=None, compare=False)

    def __str__(self) -> str:
        line = f"{self.score:7.3f}  {self.title}  ({self.path})"
        return f"{line}\n         {self.snippet}" if self.snippet else line


class ResultList(Sequence[SearchResult]):
    """The top results plus ``total`` (how many documents matched in all)."""

    def __init__(self, results: list[SearchResult], total: int, query: str) -> None:
        self._results = results
        self.total = total
        self.query = query

    @overload
    def __getitem__(self, i: int) -> SearchResult: ...
    @overload
    def __getitem__(self, i: slice) -> ResultList: ...
    def __getitem__(self, i: int | slice) -> SearchResult | ResultList:
        if isinstance(i, slice):  # results[:5] stays a ResultList, total included
            return ResultList(self._results[i], self.total, self.query)
        return self._results[i]

    def __len__(self) -> int:
        return len(self._results)

    def __iter__(self) -> Iterator[SearchResult]:
        return iter(self._results)

    def __repr__(self) -> str:
        return f"ResultList(query={self.query!r}, shown={len(self)}, total={self.total})"


class Searcher:
    """Ranked search over one index, with an LRU cache on query -> ranked doc ids."""

    def __init__(self, index: Index, cache_size: int = 256) -> None:
        self.index = index
        # a per-instance cache: lru_cache on the method itself would be shared by all
        # instances and keep every index ever searched alive
        self._ranked = functools.lru_cache(maxsize=cache_size)(self._rank)
        self._snippet = functools.lru_cache(maxsize=4 * cache_size)(self._make_snippet)

    def _make_snippet(self, doc_id: int, terms: frozenset[str]) -> Snippet:
        return make_snippet(self.index.text(doc_id), terms)

    def _rank(self, query: str, scorer: Scorer, k: int) -> tuple[int, tuple[tuple[int, float], ...]]:
        node = parse(query)
        matched = node.evaluate(self.index)
        scores: defaultdict[int, float] = defaultdict(float)
        for term in dict.fromkeys(positive_terms(node)):  # each distinct query term once, in order
            if term not in self.index:
                continue
            for posting in self.index[term]:
                if posting.doc_id in matched:
                    scores[posting.doc_id] += scorer.score(term, posting, self.index)
        if not scores:  # e.g. "NOT java": matches, but nothing to rank them by
            return len(matched), tuple((doc_id, 0.0) for doc_id in sorted(matched)[:k])
        top = heapq.nlargest(k, scores.items(), key=itemgetter(1))
        log.debug("ranked %r: %d matches, top %d", query, len(matched), len(top))
        return len(matched), tuple(top)

    @timed
    def search(
        self, query: str, scorer: Scorer | None = None, k: int = 10, *, snippets: bool = True
    ) -> ResultList:
        scorer = scorer or SCORERS["bm25"]
        total, top = self._ranked(query, scorer, k)
        log.debug("query cache: %s", self._ranked.cache_info())
        terms = frozenset(positive_terms(parse(query))) if snippets else frozenset()
        results = [
            SearchResult(
                score,
                doc_id,
                self.index.title(doc_id),
                self.index.doc_meta[doc_id].path,
                self._snippet(doc_id, terms) if snippets and self.index.texts else None,
            )
            for doc_id, score in top
        ]
        return ResultList(results, total, query)

    def cache_info(self) -> functools._CacheInfo:
        return self._ranked.cache_info()


def search(index: Index, query: str, scorer: Scorer | None = None, k: int = 10) -> ResultList:
    """One-off search (no cache shared between calls); see ``Searcher`` for repeated queries."""
    return Searcher(index).search(query, scorer, k)


def explain_score(index: Index, node: Node, doc_id: int, scorer: Scorer) -> list[tuple[str, int, int, float]]:
    """Why did this document get its score? ``(term, tf, df, contribution)`` per query term."""
    rows: list[tuple[str, int, int, float]] = []
    for term in dict.fromkeys(positive_terms(node)):
        posting = next((p for p in index.get(term, ()) if p.doc_id == doc_id), None)
        if posting is not None:
            rows.append((term, posting.tf, index.df(term), scorer.score(term, posting, index)))
    return rows
