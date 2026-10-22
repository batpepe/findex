"""Snippets: the best window of a document around the query terms, terms highlighted."""

from __future__ import annotations

import html
import re
from collections import Counter
from collections.abc import Collection
from dataclasses import dataclass

from findex.tokenize import display_text, token_spans

_UNDERLINE = re.compile(r"([*=\-^~#_])\1{3,}")  # reST heading under/overlines
_SPACE = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class Snippet:
    """Plain-text fragments; the ones that matched a query term are flagged."""

    fragments: tuple[tuple[str, bool], ...]

    def __str__(self) -> str:
        return "".join(f"**{text}**" if hit else text for text, hit in self.fragments)

    @property
    def text(self) -> str:
        """The snippet without any highlighting."""
        return "".join(text for text, _ in self.fragments)

    @property
    def highlights(self) -> list[str]:
        return [text for text, hit in self.fragments if hit]

    def html(self) -> str:
        return "".join(
            f"<mark>{html.escape(text)}</mark>" if hit else html.escape(text) for text, hit in self.fragments
        )


def _clean(text: str) -> str:
    return _SPACE.sub(" ", _UNDERLINE.sub(" ", text.replace("``", "")))


def make_snippet(text: str, terms: Collection[str], width: int = 80, max_hits: int = 500) -> Snippet:
    """Window of about ``2 * width`` characters with the most distinct query terms.

    Scans at most ``max_hits`` matches, so a frequent term in a multi-megabyte
    document does not make one snippet cost a full re-tokenization.
    """
    shown = display_text(text)
    hits: list[tuple[int, int, str]] = []
    for token, start, end in token_spans(shown):
        if token in terms:
            hits.append((start, end, token))
            if len(hits) >= max_hits:
                break
    if not hits:
        head = _clean(shown[: 2 * width]).strip()
        return Snippet(((head + (" …" if len(shown) > 2 * width else ""), False),))

    # Sliding window over the hits (sorted by offset): which centre sees the most
    # distinct query terms within +-width? Both window edges only move forward: O(n).
    in_window: Counter[str] = Counter()
    left = right = 0
    best, best_coverage = 0, 0
    for i, (centre, _, _) in enumerate(hits):
        while right < len(hits) and hits[right][0] <= centre + width:
            in_window[hits[right][2]] += 1
            right += 1
        while hits[left][0] < centre - width:
            in_window[hits[left][2]] -= 1
            if not in_window[hits[left][2]]:
                del in_window[hits[left][2]]
            left += 1
        if len(in_window) > best_coverage:  # strictly greater: ties keep the earliest window
            best, best_coverage = i, len(in_window)
    centre = hits[best][0]
    lo, hi = max(0, centre - width), min(len(shown), centre + width)
    while lo > 0 and not shown[lo - 1].isspace() and centre - lo < width + 20:  # whole words only
        lo -= 1
    while hi < len(shown) and not shown[hi].isspace() and hi - centre < width + 20:
        hi += 1

    fragments: list[tuple[str, bool]] = [("… ", False)] if lo > 0 else []
    cursor = lo
    for start, end, _ in hits:
        if start < lo or end > hi:
            continue
        fragments.append((_clean(shown[cursor:start]), False))
        fragments.append((shown[start:end], True))
        cursor = end
    tail = _clean(shown[cursor:hi])
    fragments.append((tail + (" …" if hi < len(shown) else ""), False))
    if fragments and not fragments[0][1]:
        fragments[0] = (fragments[0][0].lstrip(), False)
    return Snippet(tuple((t, hit) for t, hit in fragments if t))
