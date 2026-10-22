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

import re
import unicodedata
from collections.abc import Iterator

_APOSTROPHES = str.maketrans({"’": "'", "ʼ": "'", "‘": "'", "＇": "'"})

# Combining marks that may follow a letter when NFC has no precomposed form.
_MARKS = "̀-ͯ҃-҉᪰-᫿᷀-᷿⃐-⃿︠-︯"
_WORD = rf"\w[\w{_MARKS}]*"
WORD_RE = re.compile(rf"{_WORD}(?:'{_WORD})*")


def display_text(text: str) -> str:
    """NFC + unified apostrophes, *not* case-folded: where words are found.

    Both steps keep one character per character, so offsets found in this string
    can be shown to the user (snippets), which is not true after ``casefold``.
    """
    text = unicodedata.normalize("NFC", text)
    return text if text.isascii() else text.translate(_APOSTROPHES)


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
        token = unicodedata.normalize("NFC", match.group().casefold())
        if token.endswith("'s"):
            token = token[:-2]
        if token.strip("_"):
            yield token


def token_spans(shown: str) -> Iterator[tuple[str, int, int]]:
    """``(token, start, end)`` for a ``display_text`` string; the same tokens as ``tokenize``."""
    folded = shown.lower() if shown.isascii() else None
    for match in WORD_RE.finditer(folded if folded is not None else shown):
        token = (
            match.group() if folded is not None else unicodedata.normalize("NFC", match.group().casefold())
        )
        if token.endswith("'s"):
            token = token[:-2]
        if token.strip("_"):
            yield token, match.start(), match.end()
