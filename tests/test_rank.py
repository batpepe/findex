"""Scoring, the three ranking sanity checks, caching and snippets (Lab 3)."""

import logging

import pytest

from findex.corpus import Document
from findex.index import Index, Posting, build_index
from findex.rank import BM25, ResultList, Scorer, Searcher, SearchResult, TfIdf
from findex.snippets import make_snippet

FILLER = " ".join(f"filler{i}" for i in range(200))  # a long document, all distinct words

DOCS = [
    Document(0, "rare", "rare", "python walrus operator"),
    Document(1, "common1", "common1", "python python python"),
    Document(2, "short", "short", "gil explained"),
    Document(3, "long", "long", f"gil {FILLER}"),
    Document(4, "other", "other", "python and the event loop"),
]


@pytest.fixture(scope="module")
def index() -> Index:
    return build_index(DOCS, positions=True)


def test_scorers_satisfy_the_protocol_and_are_callable(index: Index) -> None:
    scorers: list[Scorer] = [TfIdf(), BM25()]  # pyright checks both against the Protocol here
    posting = index["walrus"][0]
    assert all(scorer.score("walrus", posting, index) > 0 for scorer in scorers)
    tfidf, bm25 = TfIdf(), BM25()
    assert tfidf("walrus", posting, index) == tfidf.score("walrus", posting, index)
    assert bm25("walrus", posting, index) == bm25.score("walrus", posting, index)


@pytest.mark.parametrize("scorer", [TfIdf(), BM25()])
def test_rare_term_outranks_common_term(index: Index, scorer: Scorer) -> None:
    results = Searcher(index).search("python OR walrus", scorer, k=5)
    assert results[0].doc_id == 0  # the only document with the rare term "walrus"


def test_bm25_saturates_term_frequency(index: Index) -> None:
    bm25 = BM25()
    doc_len = index.doc_length(0)

    def score(tf: int) -> float:
        return bm25.score("walrus", Posting(0, tf), index) if doc_len else 0.0

    assert score(2) - score(1) > 10 * (score(21) - score(20))  # the 21st occurrence adds ~nothing
    assert score(21) - score(20) < 0.01 * score(1)


def test_bm25_prefers_the_short_document_tfidf_does_not(index: Index) -> None:
    bm25 = Searcher(index).search("gil", BM25())
    assert [r.doc_id for r in bm25] == [2, 3]
    assert bm25[0].score > bm25[1].score
    tfidf = Searcher(index).search("gil", TfIdf())
    assert tfidf[0].score == tfidf[1].score  # no length normalization


def test_results_order_and_slicing(index: Index) -> None:
    results = Searcher(index).search("python", k=10)
    assert isinstance(results, ResultList)
    assert results.total == 3
    assert list(results) == sorted(results, reverse=True)  # order=True: score first
    first_two = results[:2]
    assert isinstance(first_two, ResultList)
    assert first_two.total == 3
    assert SearchResult(1.0, 7, "a", "a") < SearchResult(2.0, 1, "b", "b")


def test_repeated_query_hits_the_cache(index: Index, caplog: pytest.LogCaptureFixture) -> None:
    searcher = Searcher(index)
    with caplog.at_level(logging.DEBUG):
        searcher.search("python")
        searcher.search("python")
    info = searcher.cache_info()
    assert (info.hits, info.misses) == (1, 1)
    assert any("Searcher.search took" in r.message for r in caplog.records)


def test_snippet_highlights_terms_around_the_best_window() -> None:
    text = "Intro. " + "x " * 200 + "The event loop runs callbacks; every event matters. " + "y " * 200
    snippet = make_snippet(text, {"event", "loop"}, width=40)
    assert [t for t, hit in snippet.fragments if hit] == ["event", "loop", "event"]
    assert str(snippet).startswith("… ")
    assert "**event** **loop**" in str(snippet)
    assert snippet.html().count("<mark>") == 3


def test_snippet_without_matches_shows_the_beginning() -> None:
    snippet = make_snippet("Title\n=====\n\nSome body text.", {"absent"})
    assert str(snippet) == "Title Some body text."
