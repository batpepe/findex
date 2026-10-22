"""The query language: parser trees, operators, evaluation, phrases (Lab 3)."""

import re

import pytest

from findex.corpus import Document
from findex.index import Index, build_index
from findex.query import And, Not, Or, Phrase, QueryError, Term, explain, parse

DOCS = [
    Document(0, "0", "loop", "the event loop runs callbacks"),
    Document(1, "1", "loop2", "a loop over every event"),
    Document(2, "2", "gil", "the free-threaded build has no GIL"),
    Document(3, "3", "java", "java has threads and an event bus"),
]


@pytest.fixture(scope="module")
def index() -> Index:
    return build_index(DOCS, positions=True)


def test_precedence_and_implicit_and() -> None:
    assert parse("a OR b c") == Or(Term("a"), And(Term("b"), Term("c")))
    assert parse("a b OR c") == Or(And(Term("a"), Term("b")), Term("c"))
    assert parse("a AND b") == parse("a b") == And(Term("a"), Term("b"))
    assert parse("a b c") == And(And(Term("a"), Term("b")), Term("c"))  # left-associative


def test_not_parentheses_and_phrases() -> None:
    assert parse("python NOT java") == And(Term("python"), Not(Term("java")))
    assert parse("NOT NOT x") == Not(Not(Term("x")))
    assert parse("(a OR b) c") == And(Or(Term("a"), Term("b")), Term("c"))
    assert parse('"event loop" asyncio') == And(Phrase(("event", "loop")), Term("asyncio"))
    assert parse("free-threaded") == Phrase(("free", "threaded"))  # compound -> phrase
    assert parse("Python's or") == And(Term("python"), Term("or"))  # lower-case "or" is a word


def test_operators_build_the_same_trees_as_the_parser() -> None:
    assert Term("a") & (Term("b") | Term("c")) & ~Term("d") == parse("a (b OR c) NOT d")


@pytest.mark.parametrize(
    ("query", "message"),
    [
        ("", "empty query"),
        ("   ", "empty query"),
        ("(a OR b", "missing ')'"),
        ("a OR", "query ends"),
        ('"event loop', "unterminated quote"),
        ("a )", "unexpected ')'"),
        ("NOT", "query ends"),
    ],
)
def test_syntax_errors_say_what_is_wrong(query: str, message: str) -> None:
    with pytest.raises(QueryError, match=re.escape(message)):
        parse(query)


def test_evaluate(index: Index) -> None:
    assert parse("event").evaluate(index) == {0, 1, 3}
    assert parse("event NOT java").evaluate(index) == {0, 1}
    assert parse("NOT event").evaluate(index) == {2}
    assert parse("(java OR gil) has").evaluate(index) == {2, 3}
    assert parse("unicorn OR gil").evaluate(index) == {2}


def test_phrases_need_adjacent_positions(index: Index) -> None:
    assert parse("event loop").evaluate(index) == {0, 1}  # both words, any order
    assert parse('"event loop"').evaluate(index) == {0}  # only where they are adjacent
    assert parse('"loop event"').evaluate(index) == set()
    assert parse("free-threaded").evaluate(index) == {2}


def test_explain_round_trips_through_the_parser() -> None:
    tree = parse('a OR (b NOT "c d")')
    assert explain(tree) == '(a OR (b AND NOT "c d"))'
    assert parse(explain(tree)) == tree
