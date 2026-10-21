"""Streaming tokenizer.

Policy (documented here and in the README, tested in tests/test_tokenize.py):

1. Unicode: the text is NFC-normalized, then ``casefold()``-ed (``ß`` -> ``ss``,
   ``Σ``/``ς`` -> ``σ``), then NFC-normalized again because case folding can
   produce decomposed sequences. Typographic and Ukrainian apostrophes
   (``’`` U+2019, ``ʼ`` U+02BC, ``‘`` U+2018, ``＇`` U+FF07) become ASCII ``'``.
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

_APOSTROPHES = str.maketrans({"\u2019": "'", "\u02bc": "'", "\u2018": "'", "\uff07": "'"})

# Combining marks that may follow a letter when NFC has no precomposed form.
_MARKS = "\u0300-\u036f\u0483-\u0489\u1ab0-\u1aff\u1dc0-\u1dff\u20d0-\u20ff\ufe20-\ufe2f"
_WORD = rf"\w[\w{_MARKS}]*"
WORD_RE = re.compile(rf"{_WORD}(?:'{_WORD})*")


def normalize_text(text: str) -> str:
    """NFC + casefold + unified apostrophes: the form every token is taken from."""
    text = unicodedata.normalize("NFC", text).casefold()
    if not text.isascii():
        text = unicodedata.normalize("NFC", text.translate(_APOSTROPHES))
    return text


def tokenize(text: str) -> Iterator[str]:
    """Yield normalized tokens of ``text`` one at a time (``re.finditer``, not ``findall``)."""
    for match in WORD_RE.finditer(normalize_text(text)):
        token = match.group()
        if token.endswith("'s"):
            token = token[:-2]
        if token.strip("_"):
            yield token
