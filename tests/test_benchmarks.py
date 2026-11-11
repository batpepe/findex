"""Benchmarks (pytest-benchmark) and a speed regression guard (Lab 8).

    uv run pytest tests/test_benchmarks.py --benchmark-only       # timings table
    uv run pytest --benchmark-disable                             # CI test job: run each once

Absolute timings depend on the machine, so CI cannot fail on them; what it can
check anywhere is the *ratio* between the two engines on the same machine, in
the same run (``test_numpy_engine_stays_much_faster``). The benchmark job in CI
uploads the full timings as an artifact for comparison between commits.
"""

import random
import timeit
from collections.abc import Iterator
from pathlib import Path
from typing import cast

import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from findex.corpus import Document
from findex.index import Index, build_index
from findex.rank import ENGINES, Engine, ResultList, Searcher
from findex.store import load, save

N_DOCS = 1500


def synthetic_docs(n: int = N_DOCS, seed: int = 8) -> list[Document]:
    """Zipf-distributed words, like real text: a few terms in every document, most in few."""
    rng = random.Random(seed)
    vocabulary = [f"w{i}" for i in range(5000)]
    weights = [1 / (rank + 1) for rank in range(len(vocabulary))]
    return [
        Document(
            i, f"d{i}.txt", f"Doc {i}", " ".join(rng.choices(vocabulary, weights, k=rng.randint(50, 400)))
        )
        for i in range(n)
    ]


@pytest.fixture(scope="module")
def mapped_index(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Index]:
    path = tmp_path_factory.mktemp("bench") / "index.bin"
    save(build_index(synthetic_docs(), positions=True), path)
    index = load(path)
    yield index
    index.close()


QUERIES = {"common": "w0", "rare": "w4000", "three-terms": "w1 w7 w30", "phrase": '"w0 w1"'}


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("query", list(QUERIES))
def test_bench_search(benchmark: BenchmarkFixture, mapped_index: Index, engine: Engine, query: str) -> None:
    searcher = Searcher(mapped_index, cache_size=0, engine=engine)
    results = cast(ResultList, benchmark(searcher.search, QUERIES[query], k=10, snippets=False))
    assert results.total > 0


def test_bench_build_index(benchmark: BenchmarkFixture) -> None:
    docs = synthetic_docs(300)
    index = cast(Index, benchmark(build_index, docs, positions=True))
    assert index.num_docs == 300


def test_numpy_engine_stays_much_faster(mapped_index: Index) -> None:
    """The regression guard: if vectorization is lost (a Python loop sneaks back into
    the hot path), the NumPy engine is no longer several times faster - fail."""

    def best(engine: Engine) -> float:
        searcher = Searcher(mapped_index, cache_size=0, engine=engine)
        return min(timeit.repeat(lambda: searcher.search("w0", k=10, snippets=False), number=20, repeat=5))

    assert best("numpy") * 5 < best("python")


def test_bench_corpus_is_realistic(tmp_path: Path) -> None:
    index = build_index(synthetic_docs(200))
    assert index.df("w0") > 190  # the most common word is in (nearly) every document
    assert index.df("w4000") < 20  # a rare one in few
    save(index, tmp_path / "i.bin")
