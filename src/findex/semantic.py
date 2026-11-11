"""Semantic and hybrid search (Lab 8, M3): embeddings, brute-force cosine search, RRF.

    findex embed index.bin            # chunk every document, embed, save index.vectors.npy + friends
    findex search index.bin "how do I make my code faster" --mode semantic
    findex search index.bin "remember results of slow calls" --mode hybrid

* **Chunks, not documents.** One vector for a 40-page document is the average of
  everything it says; a passage of ~160 words is about one thing. Each document
  is split into overlapping windows (``chunk_words`` words every ``stride``);
  a document's semantic score is its *best* chunk's score.
* **Cosine as a dot product.** Every vector is L2-normalized once, at embedding
  time, so ``E @ q`` - one matrix-vector product over all chunks, in C (BLAS) -
  gives every cosine similarity at once. Brute force is exact and, at tens of
  thousands of chunks, a few milliseconds; an ANN index (HNSW) only pays off
  around a million.
* **Hybrid = Reciprocal Rank Fusion.** BM25 scores and cosines live on
  different scales, so they are not added; RRF adds ``1 / (60 + rank)`` from
  each ranking instead. No normalization, hard to get wrong.

The model is ``all-MiniLM-L6-v2`` (384 dimensions) run with ONNX Runtime and
the ``tokenizers`` library - the same network sentence-transformers runs, without
PyTorch (a ~60 MB install instead of ~1 GB, which matters in the Docker image).
The model files are downloaded from Hugging Face on first use and cached.
"""

from __future__ import annotations

import json
import logging
import os
import re
import zlib
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
import numpy.typing as npt

from findex.index import Index
from findex.snippets import Snippet
from findex.tokenize import display_text, tokenize
from findex.vector import top_k

log = logging.getLogger(__name__)

type F32 = npt.NDArray[np.float32]
type Ranked = tuple[int, tuple[tuple[int, float], ...]]

DEFAULT_MODEL = "all-MiniLM-L6-v2"
RRF_K = 60  # the constant from Cormack et al. (2009); Elasticsearch uses it too
_HF = "https://huggingface.co/sentence-transformers/{name}/resolve/{revision}/{file}"
# Pinned: a moving "main" could change the vectors under an existing index. The revision is the
# model repository's commit; model.onnx's SHA-256 is the one Hugging Face publishes (LFS), and
# tokenizer.json's was computed from the file at that revision.
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
MODEL_SHA256 = {
    "model.onnx": "6fd5d72fe4589f189f8ebc006442dbb529bb7ce38f8082112682524616046452",
    "tokenizer.json": "be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037",
}
_OPERATORS = re.compile(r'\b(?:AND|OR|NOT)\b|["()]')


class Embedder(Protocol):
    """Text in, unit-length vectors out: ``embed(texts) -> float32[len(texts), dim]``."""

    name: str
    dim: int

    def embed(self, texts: Sequence[str]) -> F32: ...


def normalize_rows(vectors: npt.NDArray[np.floating[Any]]) -> F32:
    """Scale every row to length 1, so a dot product *is* the cosine similarity."""
    rows = vectors.astype(np.float32, copy=False)
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    return (rows / np.maximum(norms, 1e-12)).astype(np.float32, copy=False)


# --- embedders -------------------------------------------------------------------------


class _Arg(Protocol):  # what we use of onnxruntime's NodeArg (the package has no type hints)
    @property
    def name(self) -> str: ...
    @property
    def shape(self) -> Sequence[int | str | None]: ...


class _Session(Protocol):  # ... of onnxruntime.InferenceSession
    def get_inputs(self) -> Sequence[_Arg]: ...
    def get_outputs(self) -> Sequence[_Arg]: ...
    def run(self, output_names: None, input_feed: dict[str, npt.NDArray[np.int64]]) -> Sequence[Any]: ...


class _Encoding(Protocol):  # ... of tokenizers.Encoding
    @property
    def ids(self) -> list[int]: ...
    @property
    def attention_mask(self) -> list[int]: ...
    @property
    def type_ids(self) -> list[int]: ...


class _Tokenizer(Protocol):  # ... of tokenizers.Tokenizer
    def enable_truncation(self, max_length: int) -> None: ...
    def enable_padding(self, pad_id: int, pad_token: str) -> None: ...
    def encode_batch(self, input: list[str]) -> list[_Encoding]: ...


class MiniLM:
    """``sentence-transformers/all-MiniLM-L6-v2``: WordPiece -> 6-layer BERT -> mean pool -> L2.

    sentence-transformers runs exactly this pipeline (Transformer, Pooling(mean),
    Normalize); here the transformer is the ONNX export from the model's own
    repository. Batches are sorted by length so padding is minimal.
    """

    name = DEFAULT_MODEL
    dim = 384
    max_tokens = 256  # the model's max_seq_length: longer passages are truncated

    def __init__(self, session: _Session, tokenizer: _Tokenizer, batch_size: int = 32) -> None:
        self.session = session
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.tokenizer.enable_truncation(max_length=self.max_tokens)
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
        self._inputs = {arg.name for arg in session.get_inputs()}
        width = session.get_outputs()[0].shape[-1]  # 384 for the real model
        self.dim = width if isinstance(width, int) else self.dim

    @classmethod
    def load(
        cls, model_dir: Path | None = None, *, download: bool = True, threads: int | None = None
    ) -> MiniLM:
        """``threads``: ONNX Runtime's intra-op threads. Its default is the number of *physical*
        cores, which was 1 on a 2-vCPU VM; batch embedding passes every usable CPU (1.8× there)."""
        import onnxruntime  # pyright: ignore[reportMissingTypeStubs]
        from tokenizers import Tokenizer

        directory = model_files(cls.name, model_dir, download=download)
        ort = cast(Any, onnxruntime)  # untyped package: the Protocols above describe what we use
        options = ort.SessionOptions()
        options.log_severity_level = 3  # errors only
        if threads:
            options.intra_op_num_threads = threads
        session = ort.InferenceSession(
            str(directory / "model.onnx"), options, providers=["CPUExecutionProvider"]
        )
        tokenizer = cast(_Tokenizer, Tokenizer.from_file(str(directory / "tokenizer.json")))
        return cls(cast(_Session, session), tokenizer)

    def embed(self, texts: Sequence[str]) -> F32:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))  # similar lengths together
        for start in range(0, len(order), self.batch_size):
            batch = order[start : start + self.batch_size]
            out[batch] = self._embed_batch([texts[i] for i in batch])
        return out

    def _embed_batch(self, texts: list[str]) -> F32:
        encodings = self.tokenizer.encode_batch(texts)
        feed = {
            "input_ids": np.array([e.ids for e in encodings], dtype=np.int64),
            "attention_mask": np.array([e.attention_mask for e in encodings], dtype=np.int64),
            "token_type_ids": np.array([e.type_ids for e in encodings], dtype=np.int64),
        }
        outputs = self.session.run(None, {k: v for k, v in feed.items() if k in self._inputs})
        hidden: F32 = np.asarray(outputs[0], dtype=np.float32)  # last_hidden_state: (batch, tokens, dim)
        return mean_pool(hidden, feed["attention_mask"])


def mean_pool(hidden: F32, attention_mask: npt.NDArray[np.int64]) -> F32:
    """Average the token vectors of each text, ignoring padding, then normalize: (b, t, d) -> (b, d)."""
    mask = attention_mask[:, :, None].astype(np.float32)  # (b, t, 1): broadcasts over d
    summed = (hidden * mask).sum(axis=1)
    counts = np.maximum(mask.sum(axis=1), np.float32(1e-9))
    return normalize_rows(summed / counts)


class HashingEmbedder:
    """Feature hashing of findex tokens into ``dim`` buckets. *Lexical*, not semantic.

    It needs no model and no network, so the tests (and CI) exercise the whole
    semantic pipeline - chunking, storage, ``E @ q``, best chunk per document,
    RRF - with it. It cannot know that "faster" and "performance" are related.
    """

    name = "hashing"

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def embed(self, texts: Sequence[str]) -> F32:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in tokenize(text):
                out[row, zlib.crc32(token.encode()) % self.dim] += 1.0  # crc32: stable across runs
        return normalize_rows(out)


def get_embedder(name: str, model_dir: Path | None = None, *, threads: int | None = None) -> Embedder:
    if name == HashingEmbedder.name:
        return HashingEmbedder()
    if name == MiniLM.name:
        return MiniLM.load(model_dir, threads=threads)
    raise ValueError(f"unknown model {name!r} (known: {DEFAULT_MODEL}, hashing)")


def model_files(name: str, model_dir: Path | None = None, *, download: bool = True) -> Path:
    """The directory holding ``model.onnx`` and ``tokenizer.json``, downloading them once."""
    base = model_dir or Path(
        os.environ.get("FINDEX_MODEL_DIR") or Path.home() / ".cache" / "findex" / "models"
    )
    directory = base / name
    wanted = {"model.onnx": "onnx/model.onnx", "tokenizer.json": "tokenizer.json"}
    missing = [local for local in wanted if not (directory / local).is_file()]
    if missing and not download:
        raise FileNotFoundError(f"model {name} not found in {directory} (run `findex embed` once online)")
    if missing:
        import hashlib

        import httpx

        revision = os.environ.get("FINDEX_MODEL_REVISION", MODEL_REVISION)
        directory.mkdir(parents=True, exist_ok=True)
        for local in missing:
            url = _HF.format(name=name, revision=revision, file=wanted[local])
            log.info("downloading %s", url)
            partial = directory / f"{local}.part"
            digest = hashlib.sha256()
            with httpx.stream("GET", url, follow_redirects=True, timeout=60) as response:
                response.raise_for_status()
                with partial.open("wb") as f:
                    for block in response.iter_bytes(1 << 20):
                        f.write(block)
                        digest.update(block)
            expected = MODEL_SHA256.get(local) if revision == MODEL_REVISION else None
            if expected is not None and digest.hexdigest() != expected:
                partial.unlink()
                raise OSError(f"{url}: SHA-256 {digest.hexdigest()} does not match the pinned {expected}")
            partial.replace(directory / local)  # atomic: a crash never leaves half a model
    return directory


# --- chunking ---------------------------------------------------------------------------

_WORDS = re.compile(r"\S+")
_UNDERLINE = re.compile(r"([*=\-^~#_])\1{3,}")  # reST heading under/overlines carry no meaning


def chunk_spans(text: str, chunk_words: int = 160, stride: int = 128) -> list[tuple[int, int]]:
    """``(start, end)`` character spans of overlapping word windows of ``display_text(text)``."""
    words = [(m.start(), m.end()) for m in _WORDS.finditer(display_text(text))]
    if not words:
        return []
    spans: list[tuple[int, int]] = []
    for first in range(0, max(1, len(words) - chunk_words + stride), stride):
        window = words[first : first + chunk_words]
        if window:
            spans.append((window[0][0], window[-1][1]))
        if first + chunk_words >= len(words):
            break
    return spans


def passage(shown: str, start: int, end: int) -> str:
    """The text that gets embedded (and shown): the span, minus markup noise."""
    return " ".join(_UNDERLINE.sub(" ", shown[start:end]).split())


# --- the vector index -------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class VectorIndex:
    """``vectors[i]`` is chunk ``i``; ``chunks[i] = (doc_id, start, end)``; grouped by doc."""

    vectors: F32  # (n_chunks, dim), unit rows, memory-mapped from .npy
    chunks: npt.NDArray[np.int32]  # (n_chunks, 3)
    model: str
    doc_ids: npt.NDArray[np.intp]  # the documents that have chunks, ascending
    doc_starts: npt.NDArray[np.intp]  # index of each such document's first chunk

    @classmethod
    def from_arrays(cls, vectors: F32, chunks: npt.NDArray[np.int32], model: str) -> VectorIndex:
        doc_col = chunks[:, 0].astype(np.intp)
        if doc_col.size and np.any(np.diff(doc_col) < 0):
            raise ValueError("chunks must be grouped by ascending doc_id")
        starts = np.flatnonzero(np.r_[True, doc_col[1:] != doc_col[:-1]]) if doc_col.size else doc_col
        return cls(vectors, chunks, model, doc_col[starts], starts)

    def __len__(self) -> int:
        return int(self.vectors.shape[0])

    @property
    def dim(self) -> int:
        return int(self.vectors.shape[1])

    def segment(self, position: int) -> slice:
        """The chunks of the ``position``-th document that has chunks."""
        end = self.doc_starts[position + 1] if position + 1 < len(self.doc_starts) else len(self)
        return slice(int(self.doc_starts[position]), int(end))


def vector_paths(index_path: Path) -> tuple[Path, Path, Path]:
    stem = index_path.with_suffix("")
    return (
        stem.with_name(stem.name + ".vectors.npy"),
        stem.with_name(stem.name + ".chunks.npy"),
        stem.with_name(stem.name + ".vectors.json"),
    )


def build_vectors(
    index: Index,
    embedder: Embedder,
    *,
    chunk_words: int = 160,
    stride: int = 128,
    batch: int = 256,
    progress: Callable[[int, int], None] | None = None,
) -> VectorIndex:
    """Chunk every stored document text and embed the chunks, ``batch`` at a time."""
    rows: list[tuple[int, int, int]] = []
    texts: list[str] = []
    for doc_id in sorted(index.doc_meta):
        shown = display_text(index.text(doc_id))
        for start, end in chunk_spans(shown, chunk_words, stride):
            rows.append((doc_id, start, end))
            texts.append(passage(shown, start, end))
    vectors = np.zeros((len(texts), embedder.dim), dtype=np.float32)
    for first in range(0, len(texts), batch):
        vectors[first : first + batch] = embedder.embed(texts[first : first + batch])
        if progress is not None:
            progress(min(first + batch, len(texts)), len(texts))
    chunks = np.array(rows, dtype=np.int32).reshape(-1, 3)
    return VectorIndex.from_arrays(vectors, chunks, embedder.name)


def save_vectors(vindex: VectorIndex, index_path: Path, **meta: object) -> None:
    vectors_path, chunks_path, meta_path = vector_paths(index_path)
    np.save(vectors_path, np.ascontiguousarray(vindex.vectors))
    np.save(chunks_path, vindex.chunks)
    info = {"model": vindex.model, "dim": vindex.dim, "chunks": len(vindex), **meta}
    meta_path.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")


def load_vectors(index_path: Path) -> VectorIndex | None:
    """The vectors saved next to ``index_path`` (memory-mapped), or ``None`` if there are none."""
    vectors_path, chunks_path, meta_path = vector_paths(index_path)
    if not (vectors_path.is_file() and chunks_path.is_file() and meta_path.is_file()):
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    vectors = np.load(vectors_path, mmap_mode="r")  # shared by every worker via the page cache
    chunks = np.load(chunks_path)
    return VectorIndex.from_arrays(vectors, chunks, meta["model"])


# --- ranking ----------------------------------------------------------------------------


def semantic_text(query: str) -> str:
    """What gets embedded: the query without the keyword operators and quotes."""
    return " ".join(_OPERATORS.sub(" ", query).split())


def semantic_rank(vindex: VectorIndex, query_vector: F32, k: int) -> tuple[Ranked, dict[int, int]]:
    """Top ``k`` documents by their best chunk's cosine, and which chunk that was."""
    sims = vindex.vectors @ query_vector  # every chunk's cosine similarity: one BLAS call
    best = np.maximum.reduceat(sims, vindex.doc_starts)  # max of each document's chunk segment
    top = top_k(vindex.doc_ids, best.astype(np.float64), k)
    position = {doc: i for i, doc in enumerate(vindex.doc_ids.tolist())}
    best_chunk: dict[int, int] = {}
    for doc_id, _ in top:  # only k segments: a Python loop is fine here
        segment = vindex.segment(position[doc_id])
        best_chunk[doc_id] = segment.start + int(np.argmax(sims[segment]))
    return (len(vindex.doc_ids), top), best_chunk


def rrf(*rankings: Sequence[int], k: int = RRF_K) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion: ``score(d) = sum over rankings of 1 / (k + rank of d)``.

    Only ranks are used, never the raw scores, so BM25 (unbounded) and cosine
    ([-1, 1]) need no normalization. A document near the top of either list
    scores well; one near the top of both scores best. Ties: lower doc id first.
    """
    fused: defaultdict[int, float] = defaultdict(float)
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            fused[doc_id] += 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda item: (-item[1], item[0]))


def chunk_snippet(index: Index, vindex: VectorIndex, chunk: int, width: int = 160) -> Snippet:
    """Semantic hits may share no word with the query: show the passage that matched instead."""
    doc_id, start, end = (int(x) for x in vindex.chunks[chunk])
    text = passage(display_text(index.text(doc_id)), start, end)
    if len(text) > 2 * width:
        cut = text.rfind(" ", 0, 2 * width)
        text = text[: cut if cut > 0 else 2 * width] + " …"
    return Snippet(((text, False),))
