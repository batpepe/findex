"""Persist the index and load it back, in two formats.

``pickle``
    One call each way and it round-trips any object graph. It is also a
    security hole: ``pickle.load`` runs whatever the file tells it to (an
    object's ``__reduce__`` can name *any* callable, e.g. ``os.system``, plus
    its arguments). That is fine for an index file this program wrote itself
    and never acceptable for a file that came over a network. ``load`` below
    therefore only unpickles files with our own extension in our own control
    flow - it is a cache format, not an exchange format.

``binary`` (default)
    A JSON header (documents, and term -> byte offset, df, total tf) followed
    by the postings as raw little-endian ``uint32`` arrays: ``doc_ids[df]``,
    ``tfs[df]`` and, with positions, all positions concatenated. Loading
    ``mmap``s the file and parses only the header; a term's postings are decoded
    from the mapped bytes the first time they are asked for, so load time does
    not grow with the size of the postings and a search touches only the
    postings of its own terms. Nothing in the file can execute code.

Layout of the binary format::

    b"FINDEX\\x00\\x01" | uint64 header length | header JSON (UTF-8) | pad to 8 | data
"""

from __future__ import annotations

import json
import mmap
import pickle
import struct
import sys
from array import array
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, Literal

from findex.index import DocMeta, InvertedIndex, Posting

type Format = Literal["binary", "pickle"]

MAGIC = b"FINDEX\x00\x01"
_U64 = struct.Struct("<Q")

if array("I").itemsize != 4 or sys.byteorder != "little":  # pragma: no cover - exotic platforms
    raise ImportError("the binary index format assumes 4-byte little-endian unsigned ints")


def save(index: InvertedIndex, path: Path, fmt: Format = "binary") -> None:
    if fmt == "pickle":
        save_pickle(index, path)
    else:
        save_binary(index, path)


def load(path: Path) -> InvertedIndex:
    """Load either format; the first bytes say which one it is."""
    with path.open("rb") as f:
        head = f.read(len(MAGIC))
    if head == MAGIC:
        return load_binary(path)
    return load_pickle(path)


# --- pickle -----------------------------------------------------------------


def save_pickle(index: InvertedIndex, path: Path) -> None:
    if not isinstance(index.postings, dict):  # a lazily loaded index: materialize first
        index = InvertedIndex(dict(index.postings), index.doc_lengths, index.doc_meta, index.has_positions)
    with path.open("wb") as f:
        pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path: Path) -> InvertedIndex:
    # SECURITY: only ever call this on index files we built ourselves - unpickling
    # attacker-controlled bytes is arbitrary code execution (see module docstring).
    with path.open("rb") as f:
        index = pickle.load(f)
    if not isinstance(index, InvertedIndex):
        raise TypeError(f"{path} does not contain a findex index")
    return index


# --- binary -----------------------------------------------------------------


def save_binary(index: InvertedIndex, path: Path) -> None:
    terms: dict[str, list[int]] = {}
    offset = 0
    for term in sorted(index.postings):  # sorted: the same index always gives the same bytes
        plist = index.postings[term]
        total_tf = sum(p.tf for p in plist)
        terms[term] = [offset, len(plist), total_tf]
        offset += 4 * (2 * len(plist) + (total_tf if index.has_positions else 0))

    header = {
        "has_positions": index.has_positions,
        "docs": [
            [doc_id, meta.path, meta.title, index.doc_lengths[doc_id]]
            for doc_id, meta in sorted(index.doc_meta.items())
        ],
        "terms": terms,
    }
    header_bytes = json.dumps(header, ensure_ascii=False, separators=(",", ":")).encode()
    padding = -(len(MAGIC) + _U64.size + len(header_bytes)) % 8

    with path.open("wb") as f:
        f.write(MAGIC)
        f.write(_U64.pack(len(header_bytes)))
        f.write(header_bytes)
        f.write(b"\0" * padding)
        for term in terms:
            plist = index.postings[term]
            array("I", (p.doc_id for p in plist)).tofile(f)
            array("I", (p.tf for p in plist)).tofile(f)
            if index.has_positions:
                array("I", (pos for p in plist for pos in p.positions)).tofile(f)


def load_binary(path: Path) -> InvertedIndex:
    with path.open("rb") as f:
        mapped = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    if mapped[: len(MAGIC)] != MAGIC:
        raise ValueError(f"{path} is not a findex binary index")
    (header_len,) = _U64.unpack_from(mapped, len(MAGIC))
    header_start = len(MAGIC) + _U64.size
    header: dict[str, Any] = json.loads(mapped[header_start : header_start + header_len])
    data_start = header_start + header_len
    data_start += -data_start % 8

    has_positions: bool = header["has_positions"]
    index = InvertedIndex(
        postings=MappedPostings(mapped, data_start, header["terms"], has_positions),
        has_positions=has_positions,
    )
    for doc_id, doc_path, title, length in header["docs"]:
        index.doc_meta[doc_id] = DocMeta(doc_path, title)
        index.doc_lengths[doc_id] = length
    return index


class MappedPostings(Mapping[str, list[Posting]]):
    """term -> postings, decoded on demand from an ``mmap``-ed binary index."""

    def __init__(
        self, mapped: mmap.mmap, data_start: int, terms: dict[str, list[int]], has_positions: bool
    ) -> None:
        self._mapped = mapped
        self._view = memoryview(mapped)
        self._data_start = data_start
        self._terms = terms
        self._has_positions = has_positions

    def __getitem__(self, term: str) -> list[Posting]:
        offset, df, total_tf = self._terms[term]  # KeyError for unknown terms, like a dict
        start = self._data_start + offset
        doc_ids = self._view[start : start + 4 * df].cast("I")
        tfs = self._view[start + 4 * df : start + 8 * df].cast("I")
        if not self._has_positions:
            return [Posting(doc_id, tf) for doc_id, tf in zip(doc_ids, tfs, strict=True)]
        flat = self._view[start + 8 * df : start + 8 * df + 4 * total_tf].cast("I")
        plist: list[Posting] = []
        cursor = 0
        for doc_id, tf in zip(doc_ids, tfs, strict=True):
            plist.append(Posting(doc_id, tf, tuple(flat[cursor : cursor + tf])))
            cursor += tf
        return plist

    def __iter__(self) -> Iterator[str]:
        return iter(self._terms)

    def __len__(self) -> int:
        return len(self._terms)

    def __contains__(self, term: object) -> bool:
        return term in self._terms

    def df(self, term: str) -> int:
        return self._terms[term][1]

    def close(self) -> None:
        self._view.release()
        self._mapped.close()
