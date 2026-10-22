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
    ``tfs[df]`` and, with positions, all positions concatenated; then every
    document's text, zlib-compressed. Loading ``mmap``s the file and parses only
    the header; a term's postings (or a document's text) are decoded from the
    mapped bytes the first time they are asked for, so load time does not grow
    with the size of the postings and a search touches only what it needs.
    Nothing in the file can execute code.

Layout of the binary format (version 2)::

    b"FINDEX\\x00\\x02" | uint64 header length | header JSON (UTF-8) | pad to 8
    | postings section | texts section
"""

from __future__ import annotations

import json
import mmap
import pickle
import struct
import sys
import zlib
from array import array
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from findex.index import CompressedTexts, DocMeta, Index, Posting, RawPostings, decode_postings
from findex.timing import timed

type Format = Literal["binary", "pickle"]

MAGIC = b"FINDEX\x00\x02"
_U64 = struct.Struct("<Q")

if array("I").itemsize != 4 or sys.byteorder != "little":  # pragma: no cover - exotic platforms
    raise ImportError("the binary index format assumes 4-byte little-endian unsigned ints")


def save(index: Index, path: Path, fmt: Format = "binary") -> None:
    if fmt == "pickle":
        save_pickle(index, path)
    else:
        save_binary(index, path)


@timed
def load(path: Path) -> Index:
    """Load either format; the first bytes say which one it is."""
    with path.open("rb") as f:
        head = f.read(len(MAGIC))
    if head == MAGIC:
        return load_binary(path)
    if head.startswith(b"FINDEX"):
        raise ValueError(f"{path}: index written by an older findex, rebuild it with `findex.index`")
    return load_pickle(path)


# --- pickle -----------------------------------------------------------------


def save_pickle(index: Index, path: Path) -> None:
    if isinstance(index.postings, MappedPostings):  # an mmap-backed index: materialize first
        texts = None
        if index.texts is not None:
            texts = CompressedTexts({doc_id: _blob(index, doc_id) for doc_id in index.doc_meta})
        index = Index(
            dict(index.items()),
            index.doc_lengths,
            index.doc_meta,
            has_positions=index.has_positions,
            texts=texts,
        )
    with path.open("wb") as f:
        pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path: Path) -> Index:
    # SECURITY: only ever call this on index files we built ourselves - unpickling
    # attacker-controlled bytes is arbitrary code execution (see module docstring).
    with path.open("rb") as f:
        index = pickle.load(f)
    if not isinstance(index, Index):
        raise TypeError(f"{path} does not contain a findex index")
    return index


# --- binary -----------------------------------------------------------------


def _blob(index: Index, doc_id: int) -> bytes:
    texts = index.texts
    if isinstance(texts, CompressedTexts):
        return texts.blobs[doc_id]
    if isinstance(texts, MappedTexts):
        return texts.blob(doc_id)
    return zlib.compress(index.text(doc_id).encode(), 6)


def _raw(index: Index, term: str) -> RawPostings:
    """A term's postings as three ``uint32`` sequences, decoding objects only if we must."""
    raw = getattr(index.postings, "raw", None)  # ArrayPostings / MappedPostings: no objects at all
    if raw is not None:
        return raw(term)
    plist = index[term]
    positions = array("I", (pos for p in plist for pos in p.positions)) if index.has_positions else None
    return array("I", (p.doc_id for p in plist)), array("I", (p.tf for p in plist)), positions


def save_binary(index: Index, path: Path) -> None:
    terms: dict[str, list[int]] = {}
    offset = 0
    for term in sorted(index):  # sorted: the same index always gives the same bytes
        df, total_tf = index.df(term), index.cf(term)
        terms[term] = [offset, df, total_tf]
        offset += 4 * (2 * df + (total_tf if index.has_positions else 0))
    postings_size = offset

    docs: list[list[Any]] = []
    text_offset = 0
    for doc_id, meta in sorted(index.doc_meta.items()):
        size = len(_blob(index, doc_id)) if index.texts is not None else 0
        docs.append([doc_id, meta.path, meta.title, index.doc_lengths[doc_id], text_offset, size])
        text_offset += size

    header = {
        "has_positions": index.has_positions,
        "has_texts": index.texts is not None,
        "postings_size": postings_size,
        "docs": docs,
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
            doc_ids, tfs, positions = _raw(index, term)
            f.write(_as_u32_bytes(doc_ids))
            f.write(_as_u32_bytes(tfs))
            if positions is not None:
                f.write(_as_u32_bytes(positions))
        if index.texts is not None:
            for doc_id, *_ in docs:
                f.write(_blob(index, doc_id))


def _as_u32_bytes(values: Sequence[int]) -> bytes | memoryview | array[int]:
    if isinstance(values, (memoryview, array)):  # already uint32 buffers: written as they are
        return values
    return array("I", values)


def load_binary(path: Path) -> Index:
    with path.open("rb") as f:
        mapped = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    if mapped[: len(MAGIC)] != MAGIC:
        raise ValueError(f"{path} is not a findex binary index")
    (header_len,) = _U64.unpack_from(mapped, len(MAGIC))
    header_start = len(MAGIC) + _U64.size
    header: dict[str, Any] = json.loads(mapped[header_start : header_start + header_len])
    data_start = header_start + header_len
    data_start += -data_start % 8

    postings = MappedPostings(mapped, data_start, header["terms"], header["has_positions"])
    texts_start = data_start + header["postings_size"]
    text_spans: dict[int, tuple[int, int]] = {}
    doc_lengths: dict[int, int] = {}
    doc_meta: dict[int, DocMeta] = {}
    for doc_id, doc_path, title, length, text_offset, text_size in header["docs"]:
        doc_meta[doc_id] = DocMeta(doc_path, title)
        doc_lengths[doc_id] = length
        text_spans[doc_id] = (texts_start + text_offset, text_size)
    texts = MappedTexts(mapped, text_spans) if header["has_texts"] else None

    index = Index(postings, doc_lengths, doc_meta, has_positions=header["has_positions"], texts=texts)
    index.on_close(mapped.close)
    return index


class MappedPostings(Mapping[str, list[Posting]]):
    """term -> postings, decoded on demand from an ``mmap``-ed binary index.

    Every call takes its *own* ``memoryview`` of the map. One memoryview shared by
    all threads was fine under the GIL and broke on free-threaded 3.14t
    ("operation forbidden on released memoryview object"): concurrent slicing of
    a single memoryview is not thread-safe there. A fresh view costs nothing.
    """

    def __init__(
        self, mapped: mmap.mmap, data_start: int, terms: dict[str, list[int]], has_positions: bool
    ) -> None:
        self._mapped = mapped
        self._data_start = data_start
        self._terms = terms
        self._has_positions = has_positions

    def raw(self, term: str) -> RawPostings:
        """Zero-copy ``uint32`` views of a term's doc ids, tfs and (if stored) positions."""
        offset, df, total_tf = self._terms[term]  # KeyError for unknown terms, like a dict
        start = self._data_start + offset
        view = memoryview(self._mapped)
        doc_ids = view[start : start + 4 * df].cast("I")
        tfs = view[start + 4 * df : start + 8 * df].cast("I")
        if not self._has_positions:
            return doc_ids, tfs, None
        return doc_ids, tfs, view[start + 8 * df : start + 8 * df + 4 * total_tf].cast("I")

    def __getitem__(self, term: str) -> list[Posting]:
        return decode_postings(*self.raw(term))

    def __iter__(self) -> Iterator[str]:
        return iter(self._terms)

    def __len__(self) -> int:
        return len(self._terms)

    def __contains__(self, term: object) -> bool:
        return term in self._terms

    def df(self, term: str) -> int:
        return self._terms[term][1]

    def cf(self, term: str) -> int:
        return self._terms[term][2]


class MappedTexts:
    """Compressed document texts read straight from the memory-mapped file."""

    def __init__(self, mapped: mmap.mmap, spans: dict[int, tuple[int, int]]) -> None:
        self._mapped = mapped
        self._spans = spans

    def blob(self, doc_id: int) -> bytes:
        start, size = self._spans[doc_id]
        return self._mapped[start : start + size]  # mmap slicing copies: safe from any thread

    def text(self, doc_id: int) -> str:
        return zlib.decompress(self.blob(doc_id)).decode()
