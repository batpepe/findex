"""Stream a corpus one document at a time.

Supported layouts under the corpus root (mixed freely, walked in sorted order
so that doc ids are stable between runs):

* a directory tree of text files (``.txt``, ``.rst``, ``.md``) - one file is one
  document;
* JSON-lines files (``.jsonl`` or ``.jsonl.gz``) - one line is one document,
  ``{"url": ..., "title": ..., "text": ...}`` (this is what ``findex crawl``
  writes in Lab 6).

Nothing here reads more than one document into memory at a time.
"""

from __future__ import annotations

import gzip
import json
import logging
import os
import re
from collections.abc import Iterator
from pathlib import Path
from typing import IO, NamedTuple

log = logging.getLogger(__name__)

TEXT_SUFFIXES = frozenset({".txt", ".rst", ".md"})


class Document(NamedTuple):
    doc_id: int
    path: str  # file path relative to the corpus root, or the page URL
    title: str
    text: str


def iter_documents(root: Path) -> Iterator[Document]:
    """Yield every document under ``root`` lazily, assigning sequential ids."""
    doc_id = 0
    for path in _walk(root):
        for rel_path, title, text in _read_source(path, root):
            yield Document(doc_id, rel_path, title, text)
            doc_id += 1


def _walk(root: Path) -> Iterator[Path]:
    """Recursive, sorted, lazy directory walk (``rglob`` order is not stable)."""
    if root.is_file():
        yield root
        return
    with os.scandir(root) as it:
        entries = sorted(it, key=lambda e: e.name)
    for entry in entries:
        if entry.name.startswith("."):
            continue
        if entry.is_dir():
            yield from _walk(Path(entry.path))
        elif _is_supported(entry.name):
            yield Path(entry.path)


def _is_supported(name: str) -> bool:
    return name.endswith((".jsonl", ".jsonl.gz")) or Path(name).suffix in TEXT_SUFFIXES


def _read_source(path: Path, root: Path) -> Iterator[tuple[str, str, str]]:
    """Yield ``(path, title, text)`` for every document stored in one file."""
    rel_path = path.relative_to(root).as_posix() if path != root else path.name
    if path.name.endswith((".jsonl", ".jsonl.gz")):
        yield from _read_jsonl(path, rel_path)
        return
    text = _read_text(path)
    yield rel_path, extract_title(text, rel_path), text


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        # One bad file must not kill a long run: keep what we can and say so.
        log.warning("%s: not valid UTF-8 (%s); undecodable bytes replaced", path, exc.reason)
        return path.read_text(encoding="utf-8", errors="replace")


def _open_lines(path: Path) -> IO[str]:
    if path.name.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open(encoding="utf-8", errors="replace")


def _read_jsonl(path: Path, rel_path: str) -> Iterator[tuple[str, str, str]]:
    with _open_lines(path) as lines:
        for lineno, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                text = record["text"]
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                log.warning("%s:%d: skipping bad record (%s)", path, lineno, exc)
                continue
            url = record.get("url") or f"{rel_path}:{lineno}"
            yield url, record.get("title") or url, text


_FIRST_LINE = re.compile(r"^[ \t]*(?:#+[ \t]+)?(\S.*?)[ \t]*$", re.MULTILINE)
_PEP_FIELD = re.compile(r"^(PEP|Title):[ \t]*(.+?)[ \t]*$", re.MULTILINE)
_RST_MARKUP = re.compile(r"``|\\ ")  # ``literal`` quotes and "\ " escapes in PEP titles
_QUOTED_NAME = re.compile(r'"([\w.]+)"')  # the text docs build quotes module names: "venv"


def extract_title(text: str, fallback: str) -> str:
    """Best-effort human title: PEP header, Markdown heading, or first non-blank line."""
    head = text[:4096]  # never scan (or split) a whole multi-megabyte document for a title
    if head.startswith("PEP:"):
        fields = dict(reversed(_PEP_FIELD.findall(head)))  # reversed: the first occurrence wins
        if "PEP" in fields and "Title" in fields:
            title = _RST_MARKUP.sub("", fields["Title"])
            return f"PEP {fields['PEP']} \N{EN DASH} {title}"
    match = _FIRST_LINE.search(head)
    if not match:
        return fallback
    return _QUOTED_NAME.sub(r"\1", match.group(1)).replace(" --- ", " \N{EM DASH} ")
