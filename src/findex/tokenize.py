"""Streaming tokenizer.

Policy (documented here and in the README, tested in tests/test_tokenize.py):

1. Unicode: the text is NFC-normalized and typographic and Ukrainian
   apostrophes (``’`` U+2019, ``ʼ`` U+02BC, ``‘`` U+2018, ``＇`` U+FF07) become
   ASCII ``'``. Words are found in *that* text; each word is then
   ``casefold()``-ed (``ß`` -> ``ss``, ``Σ``/``ς`` -> ``σ``) and NFC-normalized
   again, because case folding can produce decomposed sequences.
   (Folding *before* splitting, as Lab 1 did, lets folding change what counts
   as a word: U+0345, a combining mark, folds to the letter ``ι``. Hypothesis
   found that in Lab 4.)
2. A token is a run of Unicode word characters (``\\w``: letters and digits of
   any script, plus ``_``) together with combining marks, starting with a word
   character.
3. Apostrophes: kept *inside* a word (``don't``, ``п'ять``, ``o'clock``), never
   at its edges (``'quoted'`` -> ``quoted``). The English possessive ``'s`` is
   dropped (``python's`` -> ``python``).
4. Hyphens split words (``free-threaded`` -> ``free``, ``threaded``): a query for
   ``threaded`` should find it, and phrase search recovers the compound.
5. Digits: numbers are tokens (``2026``); a decimal point splits (``3.15`` ->
   ``3``, ``15``); mixed alphanumerics stay whole (``utf8``, ``py3``).
6. Underscores are word characters, so Python identifiers survive whole
   (``__init__``, ``snake_case``); a token made only of underscores is dropped.
7. No stop-word removal and no stemming: IDF already down-weights ``the``, and
   phrase queries need every word.
"""

from __future__ import annotations

import functools
import re
import unicodedata
from collections.abc import Collection, Iterator

_APOSTROPHES = ("’", "ʼ", "‘", "＇")  # all become ASCII '

# Combining marks that may follow a letter when NFC has no precomposed form.
_MARKS = "̀-ͯ҃-҉᪰-᫿᷀-᷿⃐-⃿︠-︯"
_WORD = rf"\w[\w{_MARKS}]*"
WORD_RE = re.compile(rf"{_WORD}(?:'{_WORD})*")
_WORDISH = re.compile(rf"[\w{_MARKS}']")  # every character a token can contain


def display_text(text: str) -> str:
    """NFC + unified apostrophes, *not* case-folded: where words are found.

    Both steps keep one character per character, so offsets found in this string
    can be shown to the user (snippets), which is not true after ``casefold``.

    Four ``str.replace`` calls, not one ``str.translate``: translate with a dict
    looks every character up in it (56 ms on the 3.4 MB changelog), replace
    searches for one character with a vectorized scan (1.4 ms for all four).
    """
    text = unicodedata.normalize("NFC", text)
    if text.isascii():
        return text
    for apostrophe in _APOSTROPHES:
        text = text.replace(apostrophe, "'")
    return text


def tokenize(text: str) -> Iterator[str]:
    """Yield normalized tokens of ``text`` one at a time (``re.finditer``, not ``findall``)."""
    shown = display_text(text)
    if shown.isascii():  # O(1) flag check; for ASCII, lower() == casefold(): fold once, in C
        for match in WORD_RE.finditer(shown.lower()):
            token = match.group()
            if token.endswith("'s"):
                token = token[:-2]
            if token.strip("_"):
                yield token
        return
    for match in WORD_RE.finditer(shown):
        word = match.group()
        # most words of a non-ASCII document are ASCII: fold those the cheap way too (Lab 8)
        token = word.lower() if word.isascii() else unicodedata.normalize("NFC", word.casefold())
        if token.endswith("'s"):
            token = token[:-2]
        if token.strip("_"):
            yield token


def token_spans(shown: str) -> Iterator[tuple[str, int, int]]:
    """``(token, start, end)`` for a ``display_text`` string; the same tokens as ``tokenize``."""
    folded = shown.lower() if shown.isascii() else None
    for match in WORD_RE.finditer(folded if folded is not None else shown):
        word = match.group()
        if folded is not None or word.isascii():
            token = word if folded is not None else word.lower()
        else:
            token = unicodedata.normalize("NFC", word.casefold())
        if token.endswith("'s"):
            token = token[:-2]
        if token.strip("_"):
            yield token, match.start(), match.end()


@functools.lru_cache(maxsize=1024)
def _alternation(terms: frozenset[str]) -> re.Pattern[str]:
    return re.compile("|".join(re.escape(t) for t in sorted(terms, key=len, reverse=True)))


def _token_at(shown: str, start: int) -> re.Match[str] | None:
    """The token ``WORD_RE.finditer(shown)`` would yield *starting at* ``start``, if any.

    A token never contains a character outside ``[\\w marks ']``, so the scan can
    restart at the last such character before ``start`` and still see exactly
    the tokens the full left-to-right scan sees.
    """
    begin = start
    while begin > 0 and _WORDISH.match(shown, begin - 1):
        begin -= 1
    for match in WORD_RE.finditer(shown, begin):
        if match.start() >= start:
            return match if match.start() == start else None
        if match.end() > start:  # ``start`` is inside an earlier token
            return None
    return None


def hit_spans(shown: str, terms: Collection[str]) -> Iterator[tuple[str, int, int]]:
    """The ``token_spans(shown)`` whose token is in ``terms`` - without tokenizing everything.

    Lab 8: the profile showed snippets re-tokenizing whole documents in Python
    (a 3 MB changelog is 500,000 tokens for ten highlights). Here one regex - the
    terms joined by ``|`` - scans a case-folded copy *in C*, and only its
    candidates are checked against the tokenizer's rules (a real token start, the
    whole token, the same normalization), so the result is exactly the same.
    The copy must keep every offset: true for ASCII, checked otherwise, and a
    text where case folding changes lengths (``ß`` -> ``ss``) takes the old path.
    """
    if not terms:
        return
    is_ascii = shown.isascii()
    folded = shown.lower() if is_ascii else shown.casefold()
    if not is_ascii and (len(folded) != len(shown) or not unicodedata.is_normalized("NFC", folded)):
        yield from (span for span in token_spans(shown) if span[0] in terms)
        return
    pattern = _alternation(frozenset(terms))
    pos = 0
    while (candidate := pattern.search(folded, pos)) is not None:
        match = _token_at(shown, candidate.start())
        if match is None:  # inside another word ("lock" in "block"): look again one further on
            pos = candidate.start() + 1
            continue
        word = match.group()
        token = word.lower() if word.isascii() else unicodedata.normalize("NFC", word.casefold())
        if token.endswith("'s"):
            token = token[:-2]
        if token in terms and token.strip("_"):
            yield token, match.start(), match.end()
        pos = match.end()  # no token starts inside this one
