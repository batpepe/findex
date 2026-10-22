"""Set operations on sorted doc-id lists (Lab 2).

The two-pointer merges are what real engines do - they stream, need no extra
memory and work on compressed or memory-mapped postings - and the same
``intersect`` finds phrase matches on positions (``findex.query``). The set
versions are the baseline they are benchmarked against (``--engine set``).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

type SetOp = Callable[[Sequence[int], Sequence[int]], list[int]]


# --- two-pointer merges over sorted doc-id lists: O(len(a) + len(b)), no hashing ---


def intersect(a: Sequence[int], b: Sequence[int]) -> list[int]:
    out: list[int] = []
    i = j = 0
    while i < len(a) and j < len(b):
        x, y = a[i], b[j]
        if x == y:
            out.append(x)
            i += 1
            j += 1
        elif x < y:
            i += 1
        else:
            j += 1
    return out


def union(a: Sequence[int], b: Sequence[int]) -> list[int]:
    out: list[int] = []
    i = j = 0
    while i < len(a) and j < len(b):
        x, y = a[i], b[j]
        if x == y:
            out.append(x)
            i += 1
            j += 1
        elif x < y:
            out.append(x)
            i += 1
        else:
            out.append(y)
            j += 1
    out.extend(a[i:])
    out.extend(b[j:])
    return out


def difference(a: Sequence[int], b: Sequence[int]) -> list[int]:
    """``a AND NOT b``: keep the ids of ``a`` that the merge does not find in ``b``."""
    out: list[int] = []
    i = j = 0
    while i < len(a):
        if j == len(b) or a[i] < b[j]:
            out.append(a[i])
            i += 1
        elif a[i] == b[j]:
            i += 1
            j += 1
        else:
            j += 1
    return out


MERGE_OPS: dict[str, SetOp] = {"AND": intersect, "OR": union, "NOT": difference}
SET_OPS: dict[str, SetOp] = {
    "AND": lambda a, b: sorted(set(a).intersection(b)),
    "OR": lambda a, b: sorted(set(a).union(b)),
    "NOT": lambda a, b: sorted(set(a).difference(b)),
}
