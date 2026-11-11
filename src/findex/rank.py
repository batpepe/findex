"""Ranking: TF-IDF and BM25 behind one ``Scorer`` protocol, top-k with ``heapq``.

``Searcher.search`` parses the query (``findex.query``), evaluates it to the set
of matching documents, scores those documents term by term with the chosen
scorer, keeps the ``k`` best with ``heapq.nlargest`` (O(n log k), not a full
sort), and attaches a highlighted snippet to each. The query -> ranked ids step
is memoized with ``functools.lru_cache``; repeated queries skip straight to
the snippets.

Two engines compute that ranking (Lab 8): ``"python"`` is the Lab 3 code below,
one ``Posting`` and one ``score()`` call at a time; ``"numpy"`` (the default) is
``findex.vector``, the same arithmetic on whole postings arrays. They return
identical rankings; ties go to the lower doc id in both.
"""

from __future__ import annotations

import functools
import heapq
import logging
import math
from collections import defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Literal, NamedTuple, Protocol, overload

import numpy as np

from findex import vector
from findex.index import I32, Index, Posting
from findex.query import Node, QueryError, parse, positive_terms
from findex.semantic import Embedder, VectorIndex, chunk_snippet, rrf, semantic_rank, semantic_text
from findex.snippets import Snippet, make_snippet
from findex.timing import timed
from findex.tokenize import tokenize
from findex.vector import F64, VectorScorer

log = logging.getLogger(__name__)

type Engine = Literal["numpy", "python"]
ENGINES: tuple[Engine, ...] = ("numpy", "python")
type Mode = Literal["keyword", "semantic", "hybrid"]
MODES: tuple[Mode, ...] = ("keyword", "semantic", "hybrid")
RRF_DEPTH = 100  # hybrid fuses this many results from each ranking


class Scorer(VectorScorer, Protocol):
    """Anything with these methods is a scorer - no inheritance needed (structural typing).

    ``score`` scores one posting (the Lab 3 engine); ``score_array`` (from
    ``VectorScorer``) scores a whole postings list at once (the NumPy engine).
    """

    def score(self, term: str, posting: Posting, index: Index) -> float: ...


_log_table: F64 = np.zeros(1)  # _log_table[tf] == math.log(tf) for tf >= 1; grown on demand


def log_tf(tfs: I32) -> F64:
    """``math.log`` of every tf, by table lookup: the exact floats Lab 3's scalar code computes.

    (``np.log`` may round differently from the C library in the last bit; a
    table filled with ``math.log`` keeps the two engines bit-for-bit identical.)
    """
    global _log_table
    table = _log_table
    top = int(tfs.max(initial=0))
    if top >= table.size:
        size = max(top + 1, 2 * table.size, 1024)
        table = np.array([0.0] + [math.log(tf) for tf in range(1, size)])
        _log_table = table  # a race only builds the same table twice
    return table[tfs]


@dataclass(frozen=True, slots=True)
class TfIdf:
    """``(1 + ln tf) * ln(N / df)``: sublinear term frequency, rare terms weigh more.

    No document-length normalization: one match in a 2 MB changelog counts as
    much as one match in a 2 KB page. That is what BM25 fixes.
    """

    name = "tfidf"

    def score(self, term: str, posting: Posting, index: Index) -> float:
        return (1 + math.log(posting.tf)) * math.log(index.num_docs / index.df(term))

    def score_array(self, term: str, doc_ids: I32, tfs: I32, index: Index) -> F64:
        return (1 + log_tf(tfs)) * math.log(index.num_docs / index.df(term))

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

    def score_array(self, term: str, doc_ids: I32, tfs: I32, index: Index) -> F64:
        # the same expression, operation for operation, on arrays: bit-identical floats
        length_norm = 1 - self.b + self.b * index.doc_len_array[doc_ids] / index.avg_doc_length
        tf = tfs.astype(np.float64)
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


def _best_first(item: tuple[int, float]) -> tuple[float, int]:
    doc_id, score = item
    return score, -doc_id  # higher score first; on a tie the lower doc id (as in findex.vector)


class SemanticUnavailable(LookupError):
    """``mode="semantic"``/``"hybrid"`` without embeddings: run ``findex embed`` first."""


type Ranking = tuple[int, tuple[tuple[int, float], ...], tuple[tuple[int, int], ...]]  # + best chunks


class Searcher:
    """Ranked search over one index, with an LRU cache on query -> ranked doc ids.

    ``mode="keyword"`` is the query language + BM25/TF-IDF; ``"semantic"`` ranks
    documents by their best chunk's cosine similarity to the embedded query;
    ``"hybrid"`` fuses the two rankings with RRF (``findex.semantic``).
    """

    def __init__(
        self,
        index: Index,
        cache_size: int = 256,
        engine: Engine = "numpy",
        *,
        vectors: VectorIndex | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        self.index = index
        self.engine = engine
        self.vectors = vectors
        self.embedder = embedder
        # a per-instance cache: lru_cache on the method itself would be shared by all
        # instances and keep every index ever searched alive
        rank = self._rank_numpy if engine == "numpy" else self._rank
        self._ranked = functools.lru_cache(maxsize=cache_size)(rank)
        self._by_meaning = functools.lru_cache(maxsize=cache_size)(self._rank_semantic)
        self._snippet = functools.lru_cache(maxsize=4 * cache_size)(self._make_snippet)

    @property
    def semantic_ready(self) -> bool:
        return self.vectors is not None and self.embedder is not None

    def _make_snippet(self, doc_id: int, terms: frozenset[str], chunk: int | None = None) -> Snippet:
        if chunk is not None and self.vectors is not None:
            # found by meaning: show the passage that matched, not the window where "the" and
            # "of" from a natural-language query happen to cluster
            return chunk_snippet(self.index, self.vectors, chunk)
        return make_snippet(self.index.text(doc_id), terms)

    def _rank_semantic(self, query: str, k: int) -> Ranking:
        if self.vectors is None or self.embedder is None:
            raise SemanticUnavailable("no embeddings for this index (run `findex embed` first)")
        query_vector = self.embedder.embed([semantic_text(query) or query])[0]
        (total, top), best_chunk = semantic_rank(self.vectors, query_vector, k)
        return total, top, tuple(best_chunk.items())

    def _rank_hybrid(self, query: str, scorer: Scorer, k: int) -> Ranking:
        depth = max(k, RRF_DEPTH)  # fuse the top RRF_DEPTH of each list, not just the top k
        _, keyword = self._ranked(query, scorer, depth)
        _, semantic, best_chunks = self._by_meaning(query, depth)
        fused = rrf([doc_id for doc_id, _ in keyword], [doc_id for doc_id, _ in semantic])
        found_by_words = {doc_id for doc_id, _ in keyword}  # these get a keyword snippet
        passages = tuple((doc_id, chunk) for doc_id, chunk in best_chunks if doc_id not in found_by_words)
        return len(fused), tuple(fused[:k]), passages

    def _rank(self, query: str, scorer: Scorer, k: int) -> tuple[int, tuple[tuple[int, float], ...]]:
        """The Lab 3 engine: Python objects and one scorer call per posting."""
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
        top = heapq.nlargest(k, scores.items(), key=_best_first)
        log.debug("ranked %r: %d matches, top %d", query, len(matched), len(top))
        return len(matched), tuple(top)

    def _rank_numpy(self, query: str, scorer: Scorer, k: int) -> tuple[int, tuple[tuple[int, float], ...]]:
        """The Lab 8 engine: masks, array expressions and ``np.argpartition`` (``findex.vector``)."""
        total, top = vector.rank(self.index, parse(query), scorer, k)
        log.debug("ranked %r: %d matches, top %d (numpy)", query, total, len(top))
        return total, top

    @timed
    def search(
        self,
        query: str,
        scorer: Scorer | None = None,
        k: int = 10,
        *,
        snippets: bool = True,
        mode: Mode = "keyword",
    ) -> ResultList:
        scorer = scorer or SCORERS["bm25"]
        best_chunks: dict[int, int] = {}
        if mode == "keyword":
            total, top = self._ranked(query, scorer, k)
            log.debug("query cache: %s", self._ranked.cache_info())
        elif mode == "semantic":
            total, top, chunks = self._by_meaning(query, k)
            best_chunks = dict(chunks)
        else:
            total, top, chunks = self._rank_hybrid(query, scorer, k)
            best_chunks = dict(chunks)
        terms = frozenset[str]()
        if snippets:
            try:
                terms = frozenset(positive_terms(parse(query)))
            except QueryError:  # free text in semantic mode need not be a valid keyword query
                terms = frozenset(tokenize(query))
        results = [
            SearchResult(
                score,
                doc_id,
                self.index.title(doc_id),
                self.index.doc_meta[doc_id].path,
                self._snippet(doc_id, terms, best_chunks.get(doc_id))
                if snippets and self.index.texts
                else None,
            )
            for doc_id, score in top
        ]
        return ResultList(results, total, query)

    def cache_info(self) -> CacheStats:
        info = self._ranked.cache_info()
        return CacheStats(info.hits, info.misses, info.currsize)


class CacheStats(NamedTuple):
    hits: int
    misses: int
    size: int


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
