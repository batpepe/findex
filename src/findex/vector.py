"""The NumPy engine (Lab 8): Boolean matching, scoring and top-k as array operations.

Lab 3's ranker walks the postings in Python: a ``Posting`` object and a scorer
call per (term, document) pair, ``frozenset`` s for the Boolean query, then
``heapq``. Here a term's postings are two ``int32`` arrays viewing the
memory-mapped index file (``np.frombuffer``: no copy, no objects), the query
becomes boolean masks over all documents, and one array expression scores
every matching document of a term at once::

    norm = 1 - b + b * doc_len[doc_ids] / avg_len              # fancy indexing
    scores[doc_ids] += idf * tf * (k1 + 1) / (tf + k1 * norm)   # ufuncs; ids are unique

``np.argpartition`` then finds the top ``k`` in O(n). The arithmetic is written
in the same order as Lab 3's scalar code, the IDF is the same ``math.log``
float, and both engines break ties the same way (higher score, then lower doc
id), so the two rankings are identical - ``tests/test_vector.py`` checks it.
"""

from __future__ import annotations

import logging
from functools import singledispatch
from typing import Protocol

import numpy as np
import numpy.typing as npt

from findex.index import I32, Index, Mask, Posting
from findex.query import And, Node, Not, Or, Phrase, Term, positive_terms

log = logging.getLogger(__name__)

type F64 = npt.NDArray[np.float64]
type Ranked = tuple[int, tuple[tuple[int, float], ...]]  # (matches in all, ((doc_id, score), ...))


class VectorScorer(Protocol):
    """A scorer that can score a whole postings list in one call."""

    def score_array(self, term: str, doc_ids: I32, tfs: I32, index: Index) -> F64: ...


# --- postings as arrays -------------------------------------------------------------------


def postings_arrays(index: Index, term: str) -> tuple[I32, I32]:
    """``(doc_ids, tfs)`` of a term; zero-copy for mmap-backed and array-backed indexes."""
    arrays = getattr(index.postings, "arrays", None)
    if arrays is not None:
        return arrays(term)
    plist: list[Posting] = index[term]  # an in-memory Lab 2 index: convert (tests, small corpora)
    doc_ids = np.fromiter((p.doc_id for p in plist), dtype=np.int32, count=len(plist))
    return doc_ids, np.fromiter((p.tf for p in plist), dtype=np.int32, count=len(plist))


def position_arrays(index: Index, term: str) -> I32:
    """Every position of the term, concatenated in doc order (``tfs`` give the run lengths)."""
    positions = getattr(index.postings, "position_arrays", None)
    if positions is not None:
        return positions(term)
    plist: list[Posting] = index[term]
    return np.fromiter((pos for p in plist for pos in p.positions), dtype=np.int32)


# --- the Boolean query as masks -----------------------------------------------------------


@singledispatch
def evaluate_mask(node: Node, index: Index) -> Mask:
    """The documents a (sub)query matches, as a boolean array indexed by doc id."""
    raise TypeError(f"cannot evaluate {type(node).__name__}")


@evaluate_mask.register
def _(node: Term, index: Index) -> Mask:
    mask = np.zeros(index.doc_id_bound, dtype=np.bool_)
    if node.term in index:
        mask[postings_arrays(index, node.term)[0]] = True
    return mask


@evaluate_mask.register
def _(node: Phrase, index: Index) -> Mask:
    if any(t not in index for t in node.terms):
        return np.zeros(index.doc_id_bound, dtype=np.bool_)
    candidates = evaluate_mask(Term(node.terms[0]), index)
    for term in node.terms[1:]:
        candidates &= evaluate_mask(Term(term), index)
    if not index.has_positions:
        log.warning("index has no positions: phrase %r matched as AND", " ".join(node.terms))
        return candidates
    if not candidates.any():
        return candidates
    # A phrase starts at p in doc d when term i is at p + i for every i. Encode each
    # (doc, position - i) as one int64 key and intersect the key sets term by term:
    # Lab 3's two-pointer merge on positions, done by np.intersect1d in C.
    starts: npt.NDArray[np.int64] | None = None
    for i, term in enumerate(node.terms):
        doc_ids, tfs = postings_arrays(index, term)
        docs = np.repeat(doc_ids, tfs)  # the doc of every position, aligned with position_arrays
        positions = position_arrays(index, term)
        keep = candidates[docs] & (positions >= i)
        keys = (docs[keep].astype(np.int64) << 32) | (positions[keep].astype(np.int64) - i)
        starts = keys if starts is None else np.intersect1d(starts, keys, assume_unique=True)
        if starts.size == 0:
            break
    mask = np.zeros(index.doc_id_bound, dtype=np.bool_)
    if starts is not None and starts.size:
        mask[(starts >> 32).astype(np.intp)] = True
    return mask


@evaluate_mask.register
def _(node: And, index: Index) -> Mask:
    if isinstance(node.right, Not):  # a AND NOT b: subtract instead of building the complement
        return evaluate_mask(node.left, index) & ~evaluate_mask(node.right.operand, index)
    if isinstance(node.left, Not):
        return evaluate_mask(node.right, index) & ~evaluate_mask(node.left.operand, index)
    return evaluate_mask(node.left, index) & evaluate_mask(node.right, index)


@evaluate_mask.register
def _(node: Or, index: Index) -> Mask:
    return evaluate_mask(node.left, index) | evaluate_mask(node.right, index)


@evaluate_mask.register
def _(node: Not, index: Index) -> Mask:
    return index.doc_mask & ~evaluate_mask(node.operand, index)


# --- scoring and top-k --------------------------------------------------------------------


def rank(index: Index, node: Node, scorer: VectorScorer, k: int) -> Ranked:
    """Lab 3's ``Searcher._rank``, vectorized: same matches, same scores, same order."""
    matched = evaluate_mask(node, index)
    total = int(np.count_nonzero(matched))
    scores = np.zeros(index.doc_id_bound, dtype=np.float64)
    scored = np.zeros(index.doc_id_bound, dtype=np.bool_)
    for term in dict.fromkeys(positive_terms(node)):  # each distinct query term once, in order
        if term not in index:
            continue
        doc_ids, tfs = postings_arrays(index, term)
        keep = matched[doc_ids]
        doc_ids, tfs = doc_ids[keep], tfs[keep]
        scores[doc_ids] += scorer.score_array(term, doc_ids, tfs, index)  # ids are unique: no np.add.at
        scored[doc_ids] = True
    candidates = np.flatnonzero(scored)
    if candidates.size == 0:  # e.g. "NOT java": matches, but nothing to rank them by
        return total, tuple((int(doc_id), 0.0) for doc_id in np.flatnonzero(matched)[:k])
    return total, top_k(candidates, scores[candidates], k)


def top_k(doc_ids: npt.NDArray[np.intp], scores: F64, k: int) -> tuple[tuple[int, float], ...]:
    """The ``k`` best ``(doc_id, score)``: higher score first, lower doc id on ties.

    ``doc_ids`` must be ascending. ``np.argpartition`` finds the k-th best score
    in O(n) without sorting everything; it picks *arbitrary* members of a tie
    at the boundary, so the tied ones are then taken in doc-id order, which
    keeps the result deterministic and equal to the Python engine's.
    """
    if scores.size > k:
        kth = scores[np.argpartition(-scores, k - 1)[k - 1]]
        above = np.flatnonzero(scores > kth)
        tied = np.flatnonzero(scores == kth)[: k - above.size]
        chosen = np.concatenate((above, tied))
    else:
        chosen = np.arange(scores.size)
    chosen = chosen[np.lexsort((doc_ids[chosen], -scores[chosen]))]  # last key is the primary one
    return tuple((int(doc_ids[i]), float(scores[i])) for i in chosen)
