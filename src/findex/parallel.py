"""Parallel index construction: map (partial indexes) + reduce (merge).

* **map** - ``build_partial(paths)`` indexes a contiguous chunk of corpus files
  and returns a ``PartialIndex`` made only of ``array('I')`` buffers, ``bytes``
  and short lists: cheap to pickle back from a worker process. Workers receive
  *paths*, never documents, and read their own files.
* **reduce** - ``merge(partials)`` concatenates every term's arrays in chunk
  order. Each partial numbers its documents from 0; the merge shifts them by
  the number of documents in the chunks before it, so the final ids are exactly
  the ones the serial ``iter_documents`` walk assigns (ranges cannot be fixed up
  front: a JSON-lines file holds an unknown number of documents). Chunks are
  contiguous runs of the sorted path list, so concatenated postings stay sorted.

The same ``build_partial``/``merge`` pair runs under a plain loop, a
``ThreadPoolExecutor`` or a ``ProcessPoolExecutor`` - only the executor changes,
so the three can be benchmarked honestly against each other.
"""

from __future__ import annotations

import logging
import os
import sys
import time
import zlib
from array import array
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Executor, ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from findex.corpus import iter_paths, read_source
from findex.index import ArrayPostings, CompressedTexts, DocMeta, Index
from findex.timing import timed
from findex.tokenize import tokenize

log = logging.getLogger(__name__)

type ExecutorName = Literal["serial", "threads", "processes"]
type TermArrays = tuple[array[int], array[int], array[int]]  # doc_ids, tfs, positions (flat)


@dataclass(slots=True)
class PartialIndex:
    """One worker's share: local doc ids 0..num_docs-1, everything in flat buffers."""

    postings: dict[str, TermArrays] = field(default_factory=dict[str, TermArrays])
    doc_lengths: array[int] = field(default_factory=lambda: array("I"))
    doc_meta: list[tuple[str, str]] = field(default_factory=list[tuple[str, str]])
    texts: list[bytes] = field(default_factory=list[bytes])

    @property
    def num_docs(self) -> int:
        return len(self.doc_lengths)


def build_partial(
    paths: list[Path], root: Path, positions: bool = True, store_texts: bool = True
) -> PartialIndex:
    """Index the documents in ``paths`` (module level, so a worker process can import it)."""
    part = PartialIndex()
    postings = part.postings
    for path in paths:
        for rel_path, title, text in read_source(path, root):
            local_id = part.num_docs
            if positions:
                term_positions: defaultdict[str, list[int]] = defaultdict(list)
                position = -1
                for position, token in enumerate(tokenize(text)):
                    term_positions[token].append(position)
                length = position + 1
                for term, offsets in term_positions.items():
                    entry = postings.get(term)
                    if entry is None:
                        entry = postings[term] = (array("I"), array("I"), array("I"))
                    entry[0].append(local_id)
                    entry[1].append(len(offsets))
                    entry[2].extend(offsets)
            else:
                counts = Counter(tokenize(text))
                length = counts.total()
                for term, tf in counts.items():
                    entry = postings.get(term)
                    if entry is None:
                        entry = postings[term] = (array("I"), array("I"), array("I"))
                    entry[0].append(local_id)
                    entry[1].append(tf)
            part.doc_lengths.append(length)
            part.doc_meta.append((rel_path, title))
            if store_texts:
                part.texts.append(zlib.compress(text.encode(), 6))
    return part


@timed
def merge(partials: Iterable[PartialIndex], *, positions: bool = True, store_texts: bool = True) -> Index:
    """Reduce: concatenate the partials' postings in chunk order, shifting doc ids."""
    postings: dict[str, TermArrays] = {}
    doc_lengths: dict[int, int] = {}
    doc_meta: dict[int, DocMeta] = {}
    texts: dict[int, bytes] = {}
    offset = 0
    for part in partials:
        for term, (doc_ids, tfs, flat) in part.postings.items():
            if offset:
                doc_ids = array("I", [doc_id + offset for doc_id in doc_ids])
            entry = postings.get(term)
            if entry is None:
                postings[term] = (doc_ids, tfs, flat)  # first chunk with this term: take its arrays
            else:
                entry[0].extend(doc_ids)
                entry[1].extend(tfs)
                entry[2].extend(flat)
        for local_id, (length, (path, title)) in enumerate(zip(part.doc_lengths, part.doc_meta, strict=True)):
            doc_lengths[offset + local_id] = length
            doc_meta[offset + local_id] = DocMeta(path, title)
        for local_id, blob in enumerate(part.texts):
            texts[offset + local_id] = blob
        offset += part.num_docs
    return Index(
        ArrayPostings(postings, positions),
        doc_lengths,
        doc_meta,
        has_positions=positions,
        texts=CompressedTexts(texts) if store_texts else None,
    )


def chunk_paths(paths: list[Path], n_chunks: int) -> list[list[Path]]:
    """Split into ``n_chunks`` contiguous runs of roughly equal total *bytes*.

    Balancing by size, not by count, matters here: one file (the changelog) is
    2.4 MB while the median document is ~15 KB.
    """
    sizes = [path.stat().st_size for path in paths]
    target = sum(sizes) / max(1, n_chunks)
    chunks: list[list[Path]] = [[]]
    filled = 0.0
    for path, size in zip(paths, sizes, strict=True):
        if filled >= target * len(chunks) and len(chunks) < n_chunks:
            chunks.append([])
        chunks[-1].append(path)
        filled += size
    return [chunk for chunk in chunks if chunk]


@dataclass(slots=True)
class BuildReport:
    """Where the time went, for the benchmark."""

    chunks: int = 0
    map_seconds: float = 0.0
    merge_seconds: float = 0.0


def build_parallel(
    root: Path,
    *,
    executor: ExecutorName = "processes",
    workers: int | None = None,
    positions: bool = True,
    store_texts: bool = True,
    chunks_per_worker: int = 4,
    on_partial: Callable[[PartialIndex], None] | None = None,
    report: BuildReport | None = None,
) -> Index:
    """Build the whole index with the chosen executor; the result does not depend on it."""
    workers = workers or os.cpu_count() or 1
    paths = list(iter_paths(root))
    n_chunks = 1 if executor == "serial" else min(len(paths), workers * chunks_per_worker) or 1
    chunks = chunk_paths(paths, n_chunks)
    log.info("%s: %d files in %d chunks, %d workers", executor, len(paths), len(chunks), workers)

    t0 = time.perf_counter()
    partials = list(_map_chunks(chunks, root, executor, workers, positions, store_texts, on_partial))
    t1 = time.perf_counter()
    index = merge(partials, positions=positions, store_texts=store_texts)
    t2 = time.perf_counter()
    if report is not None:
        report.chunks, report.map_seconds, report.merge_seconds = len(chunks), t1 - t0, t2 - t1
    return index


def _map_chunks(
    chunks: list[list[Path]],
    root: Path,
    executor: ExecutorName,
    workers: int,
    positions: bool,
    store_texts: bool,
    on_partial: Callable[[PartialIndex], None] | None,
) -> Iterator[PartialIndex]:
    """Yield partials in chunk order; a failing worker re-raises here with its traceback."""
    if executor == "serial":
        for chunk in chunks:
            part = build_partial(chunk, root, positions, store_texts)
            if on_partial:
                on_partial(part)
            yield part
        return
    pool: Executor
    pool = ThreadPoolExecutor(workers) if executor == "threads" else ProcessPoolExecutor(workers)
    with pool:
        futures = [pool.submit(build_partial, chunk, root, positions, store_texts) for chunk in chunks]
        try:
            for future in futures:  # in submission order: the merge needs chunk order
                part = future.result()  # re-raises a worker's exception (with the remote traceback)
                if on_partial:
                    on_partial(part)
                yield part
        except BaseException:
            pool.shutdown(cancel_futures=True)  # fail fast: do not wait for chunks nobody will merge
            raise


# Measured in Lab 5 on a 10-core Apple M4 (4 performance + 6 efficiency cores): processes peak at 8
# workers on the normal build, free-threaded threads at 4 (see the README); 8 is a sane ceiling.
DEFAULT_WORKERS = min(8, os.cpu_count() or 1)


def default_executor() -> ExecutorName:
    """Processes on the normal (GIL) build; threads where the GIL is disabled (3.14t)."""
    gil_enabled: Callable[[], bool] = getattr(sys, "_is_gil_enabled", lambda: True)
    return "processes" if gil_enabled() else "threads"
