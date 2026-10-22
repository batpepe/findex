"""Tokenizer: the documented policy as a parametrized table, plus corpus reading."""

import inspect
from pathlib import Path

import pytest

from findex.corpus import extract_title, iter_documents
from findex.tokenize import display_text, token_spans, tokenize


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        pytest.param("Python PYTHON pyThOn", ["python"] * 3, id="mixed-case"),
        pytest.param("Straße STRASSE", ["strasse", "strasse"], id="casefold-not-lower"),
        pytest.param(
            "Привіт, Світе! Їжак і ґанок.", ["привіт", "світе", "їжак", "і", "ґанок"], id="cyrillic"
        ),
        pytest.param("cafe\u0301", ["caf\u00e9"], id="combining-accent-nfc"),
        pytest.param(
            "Hello, world!!! (yes) -- no; ok?", ["hello", "world", "yes", "no", "ok"], id="punctuation"
        ),
        pytest.param("", [], id="empty"),
        pytest.param("   \n\t  ", [], id="whitespace"),
        pytest.param("... --- !!!", [], id="only-punctuation"),
        pytest.param("don't 'quoted' o'clock", ["don't", "quoted", "o'clock"], id="apostrophes-inside"),
        pytest.param("пʼять п’ять п'ять", ["п'ять"] * 3, id="ukrainian-apostrophes"),
        pytest.param("Python's GIL", ["python", "gil"], id="possessive"),
        pytest.param(
            "free-threaded state-of-the-art", ["free", "threaded", "state", "of", "the", "art"], id="hyphens"
        ),
        pytest.param(
            "Python 3.15 in 2026, utf8 and py3",
            ["python", "3", "15", "in", "2026", "utf8", "and", "py3"],
            id="digits",
        ),
        pytest.param(
            "__init__ snake_case ___ sys.getswitchinterval()",
            ["__init__", "snake_case", "sys", "getswitchinterval"],
            id="identifiers",
        ),
        pytest.param("\u0345", [], id="hypothesis-found-U+0345"),  # a mark that folds to a letter
    ],
)
def test_tokenize_policy(text: str, expected: list[str]) -> None:
    assert list(tokenize(text)) == expected


def test_token_spans_point_into_the_displayed_text() -> None:
    shown = display_text("The CAFE\u0301\u2019s GIL")
    spans = list(token_spans(shown))
    assert [t for t, _, _ in spans] == list(tokenize("The CAFE\u0301\u2019s GIL"))
    assert [shown[s:e] for _, s, e in spans] == ["The", "CAFÉ's", "GIL"]


def test_tokenize_and_iter_documents_are_lazy(tmp_path: Path) -> None:
    assert inspect.isgeneratorfunction(tokenize)
    assert inspect.isgeneratorfunction(iter_documents)
    (tmp_path / "a.txt").write_text("Title A\n\nbody", encoding="utf-8")
    (tmp_path / "b.txt").write_bytes(b"Title B\n\n\xff broken bytes")  # not UTF-8
    docs = list(iter_documents(tmp_path))
    assert [(d.doc_id, d.path, d.title) for d in docs] == [(0, "a.txt", "Title A"), (1, "b.txt", "Title B")]
    assert "\ufffd" in docs[1].text  # decoded with replacement, not crashed


def test_iter_documents_reads_every_format_in_sorted_order(corpus_dir: Path) -> None:
    docs = list(iter_documents(corpus_dir))
    assert [d.path for d in docs] == [
        "https://example.org/a",
        "https://example.org/b",
        "docs/asyncio.txt",
        "docs/threading.txt",
        "docs/unicode.md",
        "peps/pep-9999.rst",
    ]
    assert [d.doc_id for d in docs] == list(range(6))
    assert docs[4].title == "Unicode HOWTO"
    assert docs[5].title == "PEP 9999 \N{EN DASH} The Walrus Guide"


def test_bad_jsonl_lines_are_skipped(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    (tmp_path / "c.jsonl").write_text('{"text": "ok"}\nnot json\n{"no_text": 1}\n\n', encoding="utf-8")
    assert [d.text for d in iter_documents(tmp_path)] == ["ok"]
    assert sum("skipping bad record" in r.message for r in caplog.records) == 2


@pytest.mark.parametrize(
    ("text", "title"),
    [
        (
            '"venv" --- Creation of virtual environments\n****',
            "venv \N{EM DASH} Creation of virtual environments",
        ),
        ("PEP: 8\nTitle: Style ``Guide``\n", "PEP 8 \N{EN DASH} Style Guide"),
        ("\n\n   \n", "fallback"),
    ],
)
def test_extract_title(text: str, title: str) -> None:
    assert extract_title(text, "fallback") == title
