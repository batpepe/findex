"""Parallel indexing (Lab 5): every executor must build the very same index."""

import os
from array import array
from pathlib import Path

import pytest

from findex.corpus import iter_documents, iter_paths
from findex.index import build_index
from findex.parallel import ExecutorName, PartialIndex, build_parallel, chunk_paths, merge
from findex.store import save


@pytest.fixture(scope="module")
def reference_bytes(corpus_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> bytes:
    """The Lab 4 index (one ordered stream of documents), saved."""
    path = tmp_path_factory.mktemp("ref") / "ref.bin"
    save(build_index(iter_documents(corpus_dir), positions=True), path)
    return path.read_bytes()


@pytest.mark.parametrize(
    ("executor", "workers"),
    [("serial", 1), ("threads", 1), ("threads", 3), ("processes", 2)],
)
def test_every_executor_builds_a_byte_identical_index(
    corpus_dir: Path, reference_bytes: bytes, tmp_path: Path, executor: ExecutorName, workers: int
) -> None:
    index = build_parallel(corpus_dir, executor=executor, workers=workers, chunks_per_worker=2)
    save(index, tmp_path / "i.bin")
    assert (tmp_path / "i.bin").read_bytes() == reference_bytes


def test_without_positions_too(corpus_dir: Path, tmp_path: Path) -> None:
    save(build_index(iter_documents(corpus_dir), positions=False), tmp_path / "ref.bin")
    save(build_parallel(corpus_dir, executor="threads", workers=2, positions=False), tmp_path / "par.bin")
    assert (tmp_path / "par.bin").read_bytes() == (tmp_path / "ref.bin").read_bytes()


def test_serial_is_one_partial_merged(corpus_dir: Path) -> None:
    from findex.parallel import build_partial

    paths = list(iter_paths(corpus_dir))
    index = merge([build_partial(paths, corpus_dir)])
    assert dict(index) == dict(build_index(iter_documents(corpus_dir), positions=True))


def test_merge_shifts_local_doc_ids() -> None:
    def part(*docs: str) -> PartialIndex:
        p = PartialIndex()
        for local_id, term in enumerate(docs):
            p.postings.setdefault(term, (array("I"), array("I"), array("I")))
            p.postings[term][0].append(local_id)
            p.postings[term][1].append(1)
            p.postings[term][2].append(0)
            p.doc_lengths.append(1)
            p.doc_meta.append((f"{term}{local_id}", term))
        return p

    index = merge([part("a", "b"), part("a"), part("b", "a")], store_texts=False)
    assert [p.doc_id for p in index["a"]] == [0, 2, 4]
    assert [p.doc_id for p in index["b"]] == [1, 3]
    assert index.num_docs == 5


def test_chunks_are_contiguous_balanced_and_complete(tmp_path: Path) -> None:
    for i, size in enumerate([10, 10, 10, 500, 10, 10, 10, 10]):
        (tmp_path / f"{i}.txt").write_text("x" * size)
    paths = list(iter_paths(tmp_path))
    chunks = chunk_paths(paths, 4)
    assert [p for chunk in chunks for p in chunk] == paths  # nothing lost, order kept
    assert len(chunks) <= 4
    assert [len(c) for c in chunks] == [4, 1, 1, 2]  # the 500-byte file fills chunk 0 on its own


@pytest.mark.parametrize("executor", ["threads", "processes"])
@pytest.mark.skipif(os.geteuid() == 0, reason="root can read a chmod 000 file")
def test_a_failing_worker_fails_the_build_loudly(tmp_path: Path, executor: ExecutorName) -> None:
    for i in range(6):
        (tmp_path / f"{i}.txt").write_text(f"document {i}")
    unreadable = tmp_path / "3.txt"
    unreadable.chmod(0)
    try:
        with pytest.raises(PermissionError):
            build_parallel(tmp_path, executor=executor, workers=2, chunks_per_worker=3)
    finally:
        unreadable.chmod(0o644)
