"""The NumPy engine (Lab 8): the same matches, scores and order as Lab 3's Python engine."""

import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from findex.corpus import Document
from findex.index import Index, build_index, open_index
from findex.query import And, Node, Not, Or, Phrase, Term, explain, parse
from findex.rank import BM25, ENGINES, Engine, Searcher, TfIdf, log_tf
from findex.store import MappedPostings, load, save
from findex.vector import evaluate_mask, postings_arrays, top_k

QUERIES = [
    "the",
    "event loop",
    '"event loop"',
    "event OR walrus",
    "python NOT gil",
    "NOT the",
    '"free threaded" OR (asyncio AND NOT crawler)',
    "café",
    "absent",
    "the OR absent",
    '"loop event"',
]


def matched_ids(mask: np.ndarray) -> frozenset[int]:
    return frozenset(int(i) for i in np.flatnonzero(mask))


@pytest.mark.parametrize("query", QUERIES)
def test_masks_match_the_set_evaluation(corpus_index: Index, query: str) -> None:
    node = parse(query)
    assert matched_ids(evaluate_mask(node, corpus_index)) == node.evaluate(corpus_index)


@pytest.mark.parametrize("scorer", [BM25(), TfIdf()], ids=["bm25", "tfidf"])
@pytest.mark.parametrize("k", [1, 2, 10])
def test_both_engines_rank_identically(index_file: Path, scorer: BM25 | TfIdf, k: int) -> None:
    with open_index(index_file) as index:  # the mmap-backed index: zero-copy arrays
        python, numpy = Searcher(index, engine="python"), Searcher(index, engine="numpy")
        for query in QUERIES:
            expected = python.search(query, scorer, k, snippets=False)
            got = numpy.search(query, scorer, k, snippets=False)
            assert got.total == expected.total, query
            # same documents, same order, and the very same floats (not approximately)
            assert [(r.doc_id, r.score) for r in got] == [(r.doc_id, r.score) for r in expected], query


def test_in_memory_index_works_too(corpus_index: Index) -> None:
    python, numpy = Searcher(corpus_index, engine="python"), Searcher(corpus_index, engine="numpy")
    for query in QUERIES:
        assert [r.doc_id for r in numpy.search(query)] == [r.doc_id for r in python.search(query)]


def test_postings_are_zero_copy_views_of_the_file(index_file: Path) -> None:
    with open_index(index_file) as index:
        assert isinstance(index.postings, MappedPostings)
        doc_ids, tfs = postings_arrays(index, "event")
        assert doc_ids.dtype == np.int32 and tfs.dtype == np.int32
        assert not doc_ids.flags.owndata and not doc_ids.flags.writeable  # a read-only view of the mmap
        assert [(int(d), int(t)) for d, t in zip(doc_ids, tfs, strict=True)] == [
            (p.doc_id, p.tf) for p in index["event"]
        ]
        del doc_ids, tfs  # an exported buffer would keep the mmap from closing


def test_top_k_breaks_ties_by_doc_id() -> None:
    doc_ids = np.array([2, 5, 7, 9, 11], dtype=np.intp)
    scores = np.array([1.0, 3.0, 3.0, 3.0, 0.5])
    assert top_k(doc_ids, scores, 2) == ((5, 3.0), (7, 3.0))  # three tied at the boundary: lowest ids
    assert top_k(doc_ids, scores, 4) == ((5, 3.0), (7, 3.0), (9, 3.0), (2, 1.0))
    assert top_k(doc_ids, scores, 10) == ((5, 3.0), (7, 3.0), (9, 3.0), (2, 1.0), (11, 0.5))


def test_ties_resolve_the_same_way_in_both_engines() -> None:
    docs = [Document(i, f"d{i}", f"d{i}", "same words here") for i in range(6)]
    index = build_index(docs, positions=True)
    for engine in ENGINES:
        assert [r.doc_id for r in Searcher(index, engine=engine).search("words", k=3)] == [0, 1, 2]


def test_log_table_is_math_log_exactly() -> None:
    tfs = np.array([1, 2, 3, 17, 5000], dtype=np.int32)
    assert log_tf(tfs).tolist() == [math.log(tf) for tf in tfs.tolist()]


@pytest.mark.parametrize("engine", ENGINES)
def test_a_mapped_index_can_be_searched_from_many_threads(index_file: Path, engine: Engine) -> None:
    """Lab 7's free-threading regression test, for both engines (CI runs it on 3.14t):
    np.frombuffer takes its own buffer from the mmap on every call."""
    with open_index(index_file) as mapped:
        searcher = Searcher(mapped, cache_size=0, engine=engine)

        def search(_: int) -> list[int]:
            return [r.doc_id for r in searcher.search('"event loop" OR walrus OR the', k=5)]

        with ThreadPoolExecutor(16) as pool:
            answers = list(pool.map(search, range(400)))
    assert answers[0] and all(answer == answers[0] for answer in answers)


# --- property: random corpora, random query trees ---------------------------------------

words = st.sampled_from(["a", "b", "event", "loop", "or", "café", "the"])
texts = st.lists(st.lists(words, max_size=15).map(" ".join), min_size=1, max_size=8)
trees: st.SearchStrategy[Node] = st.recursive(
    st.one_of(words.map(Term), st.lists(words, min_size=2, max_size=3).map(lambda ws: Phrase(tuple(ws)))),
    lambda children: st.one_of(
        st.tuples(children, children).map(lambda lr: And(*lr)),
        st.tuples(children, children).map(lambda lr: Or(*lr)),
        children.map(Not),
    ),
    max_leaves=6,
)


@settings(max_examples=60, deadline=None)
@given(texts, trees, st.sampled_from([1, 3, 10]))
def test_engines_agree_on_random_corpora_and_queries(
    tmp_path_factory: pytest.TempPathFactory, corpus: list[str], tree: Node, k: int
) -> None:
    index = build_index([Document(i, f"d{i}", f"d{i}", t) for i, t in enumerate(corpus)], positions=True)
    path = tmp_path_factory.mktemp("prop") / "i.bin"
    save(index, path)
    mapped = load(path)
    try:
        assert matched_ids(evaluate_mask(tree, mapped)) == tree.evaluate(index)
        python, numpy = Searcher(index, engine="python"), Searcher(mapped, engine="numpy")
        for scorer in (BM25(), TfIdf()):
            query = explain(tree)
            expected = python.search(query, scorer, k, snippets=False)
            got = numpy.search(query, scorer, k, snippets=False)
            assert [(r.doc_id, r.score) for r in got] == [(r.doc_id, r.score) for r in expected]
    finally:
        mapped.close()
