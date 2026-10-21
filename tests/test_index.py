"""Inverted index, Boolean merges and persistence (Lab 2)."""

import random
from pathlib import Path

import pytest

from findex.corpus import Document
from findex.index import DocMeta, InvertedIndex, Posting, build_index
from findex.search import SET_OPS, boolean_search, difference, intersect, union
from findex.store import MappedPostings, load, save

DOCS = [
    Document(0, "a.txt", "A", "the cat sat on the mat"),
    Document(1, "b.txt", "B", "the dog sat"),
    Document(2, "c.txt", "C", "a cat and a dog"),
]


@pytest.fixture
def index() -> InvertedIndex:
    return build_index(DOCS, positions=True)


def test_posting_and_docmeta_are_hashable_value_objects() -> None:
    assert {Posting(1, 2), Posting(1, 2), Posting(1, 3)} == {Posting(1, 2), Posting(1, 3)}
    assert len({DocMeta("a", "A"), DocMeta("a", "A")}) == 1
    with pytest.raises(AttributeError):
        Posting(1, 2).tf = 5  # type: ignore[misc]  # frozen on purpose
    assert not hasattr(Posting(1, 2), "__dict__")  # slots: no per-instance dict


def test_build_postings_sorted_with_tf_and_positions(index: InvertedIndex) -> None:
    assert index.postings["the"] == [Posting(0, 2, (0, 4)), Posting(1, 1, (0,))]
    assert index.postings["cat"] == [Posting(0, 1, (1,)), Posting(2, 1, (1,))]
    assert index.doc_lengths == {0: 6, 1: 3, 2: 5}
    assert index.doc_meta[2] == DocMeta("c.txt", "C")
    for plist in index.postings.values():
        ids = [p.doc_id for p in plist]
        assert ids == sorted(ids)


def test_merges_agree_with_python_sets() -> None:
    rng = random.Random(42)
    for _ in range(200):
        a = sorted(rng.sample(range(100), rng.randint(0, 40)))
        b = sorted(rng.sample(range(100), rng.randint(0, 40)))
        assert intersect(a, b) == SET_OPS["AND"](a, b)
        assert union(a, b) == SET_OPS["OR"](a, b)
        assert difference(a, b) == SET_OPS["NOT"](a, b)


@pytest.mark.parametrize("engine", ["merge", "set"])
def test_boolean_queries(index: InvertedIndex, engine: str) -> None:
    assert boolean_search(index, "cat", engine) == [0, 2]
    assert boolean_search(index, "cat dog", engine) == [2]
    assert boolean_search(index, "cat OR dog", engine) == [0, 1, 2]
    assert boolean_search(index, "sat NOT dog", engine) == [0]
    assert boolean_search(index, "NOT cat", engine) == [1]
    assert boolean_search(index, "THE Cat's", engine) == [0]
    assert boolean_search(index, "unicorn", engine) == []


@pytest.mark.parametrize("fmt", ["binary", "pickle"])
def test_save_load_round_trip(index: InvertedIndex, tmp_path: Path, fmt: str) -> None:
    path = tmp_path / f"index.{fmt}"
    save(index, path, fmt)
    loaded = load(path)
    assert dict(loaded.postings) == dict(index.postings)
    assert loaded.doc_lengths == index.doc_lengths
    assert loaded.doc_meta == index.doc_meta
    assert loaded.has_positions
    if fmt == "binary":
        assert isinstance(loaded.postings, MappedPostings)
        assert "cat" in loaded.postings
        assert "unicorn" not in loaded.postings
        with pytest.raises(KeyError):
            loaded.postings["unicorn"]
        loaded.postings.close()


def test_binary_format_is_deterministic(index: InvertedIndex, tmp_path: Path) -> None:
    save(index, tmp_path / "1.bin")
    save(load(tmp_path / "1.bin"), tmp_path / "2.bin")
    assert (tmp_path / "1.bin").read_bytes() == (tmp_path / "2.bin").read_bytes()
