"""Property-based tests (Hypothesis): invariants, oracles and round trips."""

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from findex.corpus import Document
from findex.index import build_index
from findex.merge import difference, intersect, union
from findex.query import And, Node, Not, Or, Phrase, Term, explain, parse
from findex.store import load, save
from findex.tokenize import display_text, token_spans, tokenize

sorted_ids = st.lists(st.integers(min_value=0, max_value=500), unique=True).map(sorted)


@pytest.mark.slow  # ~1 s: 100 random texts through the tokenizer twice
@given(st.text())
def test_tokenize_is_idempotent(text: str) -> None:
    tokens = list(tokenize(text))
    assert list(tokenize(" ".join(tokens))) == tokens


@given(st.text())
def test_tokens_are_never_empty_or_padded(text: str) -> None:
    for token in tokenize(text):
        assert token
        assert token == token.strip()
        assert token.strip("_")


@given(st.text())
def test_token_spans_agree_with_tokenize(text: str) -> None:
    assert [t for t, _, _ in token_spans(display_text(text))] == list(tokenize(text))


@given(sorted_ids, sorted_ids)
def test_merges_agree_with_the_set_oracle(a: list[int], b: list[int]) -> None:
    assert intersect(a, b) == sorted(set(a) & set(b))
    assert union(a, b) == sorted(set(a) | set(b))
    assert difference(a, b) == sorted(set(a) - set(b))


@given(sorted_ids, sorted_ids)
def test_merge_results_stay_sorted_and_unique(a: list[int], b: list[int]) -> None:
    for merged in (intersect(a, b), union(a, b), difference(a, b)):
        assert merged == sorted(set(merged))


words = st.sampled_from(["the", "event", "loop", "gil", "python", "café", "п'ять", "walrus", "x"])
documents = st.lists(st.lists(words, max_size=12).map(" ".join), min_size=1, max_size=8)


@settings(max_examples=50, deadline=None)
@given(documents, st.booleans())
def test_save_load_round_trip(texts: list[str], positions: bool) -> None:
    docs = [Document(i, f"d{i}.txt", f"Doc {i}", text) for i, text in enumerate(texts)]
    index = build_index(docs, positions=positions)
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "index.bin"
        save(index, path)
        loaded = load(path)
        assert dict(loaded) == dict(index)
        assert loaded.doc_lengths == index.doc_lengths
        assert loaded.doc_meta == index.doc_meta
        assert [loaded.text(d.doc_id) for d in docs] == texts
        loaded.close()


query_words = st.sampled_from(["a", "b", "event", "loop", "or", "and", "café"])  # lower-case or/and are words
query_trees: st.SearchStrategy[Node] = st.recursive(
    st.one_of(
        query_words.map(Term),
        st.lists(query_words, min_size=2, max_size=3).map(lambda ws: Phrase(tuple(ws))),
    ),
    lambda children: st.one_of(
        st.tuples(children, children).map(lambda lr: And(*lr)),
        st.tuples(children, children).map(lambda lr: Or(*lr)),
        children.map(Not),
    ),
    max_leaves=8,
)


@given(query_trees)
def test_explain_then_parse_gives_back_the_same_tree(tree: Node) -> None:
    assert parse(explain(tree)) == tree
