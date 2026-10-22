"""The query language: a recursive-descent parser producing a tree of nodes.

Grammar (``OR`` binds loosest, ``NOT`` tightest; ``AND`` may be omitted)::

    or_expr  := and_expr ("OR" and_expr)*
    and_expr := not_expr (["AND"] not_expr)*
    not_expr := "NOT" not_expr | atom
    atom     := WORD | "quoted phrase" | "(" or_expr ")"

so ``a OR b c`` is ``Or(Term('a'), And(Term('b'), Term('c')))``. Only upper-case
``AND``/``OR``/``NOT`` are operators; ``or`` is an ordinary word. Words and
phrases go through the document tokenizer, so a hyphenated word such as
``free-threaded`` becomes the phrase ``"free threaded"``.

Nodes are frozen dataclasses (so ``==`` compares trees, which is how the tests
are written) and overload ``&``, ``|`` and ``~``, so queries can be built in
code without any string: ``Term("python") & ~Term("java")``.
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from functools import singledispatch

from findex.index import Index
from findex.merge import intersect
from findex.tokenize import tokenize

log = logging.getLogger(__name__)


class QueryError(ValueError):
    """A query that cannot be parsed; the message says where and why."""


class Node(ABC):
    __slots__ = ()  # so the slotted dataclass nodes really have no __dict__

    def __and__(self, other: Node) -> Node:
        return And(self, other)

    def __or__(self, other: Node) -> Node:
        return Or(self, other)

    def __invert__(self) -> Node:
        return Not(self)

    @abstractmethod
    def evaluate(self, index: Index) -> frozenset[int]:
        """The ids of the documents this (sub)query matches."""


@dataclass(frozen=True, slots=True)
class Term(Node):
    term: str

    def evaluate(self, index: Index) -> frozenset[int]:
        if self.term not in index:
            return frozenset()
        return frozenset(p.doc_id for p in index[self.term])


@dataclass(frozen=True, slots=True)
class Phrase(Node):
    terms: tuple[str, ...]

    def evaluate(self, index: Index) -> frozenset[int]:
        if any(t not in index for t in self.terms):
            return frozenset()
        by_doc = [{p.doc_id: p for p in index[t]} for t in self.terms]
        candidates = frozenset(by_doc[0]).intersection(*by_doc[1:])
        if not index.has_positions:
            log.warning("index has no positions: phrase %r matched as AND", " ".join(self.terms))
            return candidates
        return frozenset(d for d in candidates if _phrase_in_doc([pd[d].positions for pd in by_doc]))


def _phrase_in_doc(positions: list[tuple[int, ...]]) -> bool:
    """Is there a start ``p`` with term ``i`` at ``p + i`` for every ``i``? Two-pointer merges."""
    starts: list[int] = list(positions[0])
    for i, pos in enumerate(positions[1:], start=1):
        starts = intersect(starts, [p - i for p in pos])
        if not starts:
            return False
    return True


@dataclass(frozen=True, slots=True)
class And(Node):
    left: Node
    right: Node

    def evaluate(self, index: Index) -> frozenset[int]:
        # sets, not the Lab 2 merge: in CPython the hashing runs in C and won the benchmark
        if isinstance(self.right, Not):  # a AND NOT b: subtract instead of building the complement
            return self.left.evaluate(index) - self.right.operand.evaluate(index)
        if isinstance(self.left, Not):
            return self.right.evaluate(index) - self.left.operand.evaluate(index)
        return self.left.evaluate(index) & self.right.evaluate(index)


@dataclass(frozen=True, slots=True)
class Or(Node):
    left: Node
    right: Node

    def evaluate(self, index: Index) -> frozenset[int]:
        return self.left.evaluate(index) | self.right.evaluate(index)


@dataclass(frozen=True, slots=True)
class Not(Node):
    operand: Node

    def evaluate(self, index: Index) -> frozenset[int]:
        return index.all_doc_ids - self.operand.evaluate(index)


def positive_terms(node: Node) -> Iterator[str]:
    """Terms that should contribute to the score: everything not under a NOT."""
    match node:
        case Term(term):
            yield term
        case Phrase(terms):
            yield from terms
        case And(left, right) | Or(left, right):
            yield from positive_terms(left)
            yield from positive_terms(right)
        case _:  # Not: excluded terms never add to a score
            return


# --- lexer + parser ---------------------------------------------------------

_LEX = re.compile(r'(?P<paren>[()])|"(?P<phrase>[^"]*)(?P<closed>"?)|(?P<word>[^\s()"]+)')
_OPERATORS = frozenset({"AND", "OR", "NOT"})

type Token = tuple[str, str | Node, int]  # (kind, value, offset in the query)


def _lex(query: str) -> list[Token]:
    tokens: list[Token] = []
    pos = 0
    while True:
        while pos < len(query) and query[pos].isspace():
            pos += 1
        if pos == len(query):
            return tokens
        m = _LEX.match(query, pos)  # always matches: any non-space character starts a token
        assert m is not None
        if paren := m["paren"]:
            tokens.append((paren, paren, pos))
        elif m["phrase"] is not None:
            if not m["closed"]:
                raise QueryError(f"unterminated quote at offset {pos}")
            if node := _text_node(m["phrase"]):
                tokens.append(("atom", node, pos))
        elif m["word"] in _OPERATORS:
            tokens.append((m["word"], m["word"], pos))
        elif node := _text_node(m["word"]):
            tokens.append(("atom", node, pos))
        pos = m.end()


def _text_node(text: str) -> Node | None:
    terms = tuple(tokenize(text))
    if not terms:
        return None  # pure punctuation: nothing to search for
    if len(terms) == 1:
        return Term(terms[0])
    return Phrase(terms)  # quoted text, or a compound such as free-threaded or sys.argv


class _Parser:
    def __init__(self, tokens: list[Token]) -> None:
        self.tokens = tokens
        self.i = 0

    def peek(self) -> str | None:
        return self.tokens[self.i][0] if self.i < len(self.tokens) else None

    def take(self) -> Token:
        token = self.tokens[self.i]
        self.i += 1
        return token

    def or_expr(self) -> Node:
        node = self.and_expr()
        while self.peek() == "OR":
            self.take()
            node = Or(node, self.and_expr())
        return node

    def and_expr(self) -> Node:
        node = self.not_expr()
        while self.peek() in ("AND", "NOT", "atom", "("):
            if self.peek() == "AND":
                self.take()
            node = And(node, self.not_expr())
        return node

    def not_expr(self) -> Node:
        if self.peek() == "NOT":
            self.take()
            return Not(self.not_expr())
        return self.atom()

    def atom(self) -> Node:
        if self.peek() is None:
            raise QueryError("query ends where a term was expected")
        kind, value, offset = self.take()
        if kind == "atom":
            assert isinstance(value, Node)
            return value
        if kind == "(":
            node = self.or_expr()
            if self.peek() != ")":
                raise QueryError(f"missing ')' for '(' at offset {offset}")
            self.take()
            return node
        raise QueryError(f"unexpected {value!r} at offset {offset}")


def parse(query: str) -> Node:
    """Parse a query string into a tree; raises ``QueryError`` on bad syntax."""
    parser = _Parser(_lex(query))
    if parser.peek() is None:
        raise QueryError("empty query")
    node = parser.or_expr()
    if parser.peek() is not None:
        _, value, offset = parser.take()
        raise QueryError(f"unexpected {value!r} at offset {offset}")
    return node


# --- explain: one function, one implementation per node type ----------------


@singledispatch
def explain(node: Node) -> str:
    raise TypeError(f"cannot explain {type(node).__name__}")


@explain.register
def _(node: Term) -> str:
    return node.term


@explain.register
def _(node: Phrase) -> str:
    return '"' + " ".join(node.terms) + '"'


@explain.register
def _(node: And) -> str:
    return f"({explain(node.left)} AND {explain(node.right)})"


@explain.register
def _(node: Or) -> str:
    return f"({explain(node.left)} OR {explain(node.right)})"


@explain.register
def _(node: Not) -> str:
    return f"NOT {explain(node.operand)}"
