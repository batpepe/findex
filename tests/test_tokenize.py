"""Tokenizer checks (plain asserts for Lab 1; run with ``uv run pytest``)."""

import inspect
from pathlib import Path

from findex.corpus import iter_documents
from findex.tokenize import tokenize


def toks(text: str) -> list[str]:
    return list(tokenize(text))


def test_mixed_case_and_casefold() -> None:
    assert toks("Python PYTHON pyThOn") == ["python"] * 3
    assert toks("Straße STRASSE") == ["strasse", "strasse"]  # casefold, not lower


def test_cyrillic() -> None:
    assert toks("Привіт, Світе! Їжак і ґанок.") == ["привіт", "світе", "їжак", "і", "ґанок"]


def test_combining_mark_accent_is_nfc_normalized() -> None:
    decomposed = "cafe\u0301"  # 'e' + COMBINING ACUTE ACCENT
    assert toks(decomposed) == toks("café") == ["café"]


def test_punctuation_is_dropped() -> None:
    assert toks("Hello, world!!! (yes) -- no; ok?") == ["hello", "world", "yes", "no", "ok"]


def test_empty_and_whitespace() -> None:
    assert toks("") == []
    assert toks("   \n\t  ") == []
    assert toks("... --- !!!") == []


def test_apostrophes_inside_words() -> None:
    assert toks("don't 'quoted' o'clock") == ["don't", "quoted", "o'clock"]
    # Ukrainian apostrophe U+02BC and typographic U+2019 both become ASCII "'"
    assert toks("пʼять п’ять п'ять") == ["п'ять"] * 3
    assert toks("Python's GIL") == ["python", "gil"]


def test_hyphens_split() -> None:
    assert toks("free-threaded state-of-the-art") == ["free", "threaded", "state", "of", "the", "art"]


def test_digits() -> None:
    assert toks("Python 3.15 in 2026, utf8 and py3") == [
        "python",
        "3",
        "15",
        "in",
        "2026",
        "utf8",
        "and",
        "py3",
    ]


def test_python_identifiers_survive() -> None:
    assert toks("__init__ snake_case ___ sys.getswitchinterval()") == [
        "__init__",
        "snake_case",
        "sys",
        "getswitchinterval",
    ]


def test_tokenize_and_iter_documents_are_lazy(tmp_path: Path) -> None:
    assert inspect.isgeneratorfunction(tokenize)
    assert inspect.isgeneratorfunction(iter_documents)
    (tmp_path / "a.txt").write_text("Title A\n\nbody", encoding="utf-8")
    (tmp_path / "b.txt").write_bytes(b"Title B\n\n\xff broken bytes")  # not UTF-8
    docs = list(iter_documents(tmp_path))
    assert [(d.doc_id, d.path, d.title) for d in docs] == [(0, "a.txt", "Title A"), (1, "b.txt", "Title B")]
    assert "\ufffd" in docs[1].text  # decoded with replacement, not crashed
