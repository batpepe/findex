"""Inverted index, Boolean merges and persistence (Lab 2), Index data model (Lab 3)."""

import random
from pathlib import Path

import pytest

from findex.corpus import Document
from findex.index import DocMeta, Index, Posting, build_index, open_index
from findex.merge import SET_OPS, difference, intersect, union
from findex.search import Engine, boolean_search
from findex.store import Format, MappedPostings, load, save

DOCS = [
    Document(0, "a.txt", "A", "the cat sat on the mat"),
    Document(1, "b.txt", "B", "the dog sat"),
    Document(2, "c.txt", "C", "a cat and a dog"),
]


@pytest.fixture
def index() -> Index:
    return build_index(DOCS, positions=True)


def test_posting_and_docmeta_are_hashable_value_objects() -> None:
    assert {Posting(1, 2), Posting(1, 2), Posting(1, 3)} == {Posting(1, 2), Posting(1, 3)}
    assert len({DocMeta("a", "A"), DocMeta("a", "A")}) == 1
    with pytest.raises(AttributeError):
        Posting(1, 2).tf = 5  # type: ignore[misc]  # frozen on purpose
    assert not hasattr(Posting(1, 2), "__dict__")  # slots: no per-instance dict


def test_build_postings_sorted_with_tf_and_positions(index: Index) -> None:
    assert index["the"] == [Posting(0, 2, (0, 4)), Posting(1, 1, (0,))]
    assert index["cat"] == [Posting(0, 1, (1,)), Posting(2, 1, (1,))]
    assert index.doc_lengths == {0: 6, 1: 3, 2: 5}
    assert index.doc_meta[2] == DocMeta("c.txt", "C")
    for plist in index.values():
        ids = [p.doc_id for p in plist]
        assert ids == sorted(ids)


def test_index_speaks_the_data_model(index: Index) -> None:
    assert len(index) == 8  # distinct terms
    assert "cat" in index
    assert "unicorn" not in index
    with pytest.raises(KeyError):
        index["unicorn"]
    assert sorted(index) == ["a", "and", "cat", "dog", "mat", "on", "sat", "the"]
    assert repr(index) == "Index(terms=8, docs=3, positions=True)"
    assert index.num_docs == 3
    assert index.avg_doc_length == pytest.approx(14 / 3)
    assert index.df("the") == 2
    assert index.df("unicorn") == 0
    assert index.text(1) == "the dog sat"


def test_merges_agree_with_python_sets() -> None:
    rng = random.Random(42)
    for _ in range(200):
        a = sorted(rng.sample(range(100), rng.randint(0, 40)))
        b = sorted(rng.sample(range(100), rng.randint(0, 40)))
        assert intersect(a, b) == SET_OPS["AND"](a, b)
        assert union(a, b) == SET_OPS["OR"](a, b)
        assert difference(a, b) == SET_OPS["NOT"](a, b)


@pytest.mark.parametrize("engine", ["merge", "set"])
def test_boolean_queries(index: Index, engine: Engine) -> None:
    assert boolean_search(index, "cat", engine) == [0, 2]
    assert boolean_search(index, "cat dog", engine) == [2]
    assert boolean_search(index, "cat OR dog", engine) == [0, 1, 2]
    assert boolean_search(index, "sat NOT dog", engine) == [0]
    assert boolean_search(index, "NOT cat", engine) == [1]
    assert boolean_search(index, "THE Cat's", engine) == [0]
    assert boolean_search(index, "unicorn", engine) == []


@pytest.mark.parametrize("fmt", ["binary", "pickle"])
def test_save_load_round_trip(index: Index, tmp_path: Path, fmt: Format) -> None:
    path = tmp_path / f"index.{fmt}"
    save(index, path, fmt)
    with open_index(path) as loaded:
        assert dict(loaded) == dict(index)
        assert loaded.doc_lengths == index.doc_lengths
        assert loaded.doc_meta == index.doc_meta
        assert loaded.has_positions
        assert [loaded.text(d) for d in range(3)] == [d.text for d in DOCS]
        if fmt == "binary":
            assert isinstance(loaded.postings, MappedPostings)
            assert "unicorn" not in loaded
            with pytest.raises(KeyError):
                loaded["unicorn"]


def test_open_index_closes_even_when_the_body_raises(index: Index, tmp_path: Path) -> None:
    save(index, tmp_path / "i.bin")
    with pytest.raises(RuntimeError), open_index(tmp_path / "i.bin") as loaded:
        raise RuntimeError("boom")
    with pytest.raises(ValueError):  # the memory map is gone: reading it now fails loudly
        loaded["cat"]


def test_binary_format_is_deterministic(index: Index, tmp_path: Path) -> None:
    save(index, tmp_path / "1.bin")
    save(load(tmp_path / "1.bin"), tmp_path / "2.bin")
    assert (tmp_path / "1.bin").read_bytes() == (tmp_path / "2.bin").read_bytes()


def test_a_mapped_index_can_be_searched_from_many_threads(index_file: Path) -> None:
    """Regression: on free-threaded 3.14t one memoryview shared by all threads broke
    ("operation forbidden on released memoryview object"). CI runs this on 3.14t."""
    from concurrent.futures import ThreadPoolExecutor

    from findex.rank import Searcher

    with open_index(index_file) as mapped:
        searcher = Searcher(mapped, cache_size=0)  # no cache: every call decodes postings again

        def search(_: int) -> list[int]:
            return [r.doc_id for r in searcher.search('"event loop" OR walrus', k=5)]

        with ThreadPoolExecutor(16) as pool:
            answers = list(pool.map(search, range(400)))
    assert answers[0] and all(answer == answers[0] for answer in answers)
