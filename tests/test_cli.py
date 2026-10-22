"""The ``findex`` command, driven through typer's CliRunner (no subprocess)."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from findex import __version__
from findex.cli import app

runner = CliRunner()


def test_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == f"findex {__version__}"


def test_index_command_builds_a_loadable_index(corpus_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "i.bin"
    result = runner.invoke(app, ["index", str(corpus_dir), "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert out.stat().st_size > 0
    assert "indexed 6 documents" in result.stderr
    assert result.stdout == ""  # diagnostics never pollute stdout


def test_index_command_rejects_a_missing_corpus(tmp_path: Path) -> None:
    result = runner.invoke(app, ["index", str(tmp_path / "nope"), "--out", str(tmp_path / "i.bin")])
    assert result.exit_code == 1
    assert "corpus not found" in result.stderr


def test_search_prints_a_ranked_table(index_file: Path) -> None:
    result = runner.invoke(app, ["search", str(index_file), "event loop", "-k", "2"])
    assert result.exit_code == 0, result.output
    assert "matches for 'event loop'" in result.stdout
    assert "Coroutines and tasks" in result.stdout


def test_search_json_lines_pipe_cleanly(index_file: Path) -> None:
    result = runner.invoke(app, ["search", str(index_file), '"event loop"', "--json"])
    assert result.exit_code == 0, result.output
    records = [json.loads(line) for line in result.stdout.splitlines()]
    assert [r["rank"] for r in records] == list(range(1, len(records) + 1))
    assert {r["path"] for r in records} == {"docs/asyncio.txt", "https://example.org/a"}
    assert all(set(r["highlights"]) <= {"event", "loop"} for r in records)


@pytest.mark.parametrize("scorer", ["bm25", "tfidf"])
def test_search_accepts_both_scorers(index_file: Path, scorer: str) -> None:
    result = runner.invoke(app, ["search", str(index_file), "gil", "--scorer", scorer, "--json"])
    assert result.exit_code == 0
    assert len(result.stdout.splitlines()) == 2


def test_bad_query_is_one_line_and_exit_code_2(index_file: Path) -> None:
    result = runner.invoke(app, ["search", str(index_file), "(event OR"])
    assert result.exit_code == 2
    assert result.stderr.strip().splitlines() == ["error: bad query: query ends where a term was expected"]


def test_missing_index_is_one_line_and_exit_code_1(tmp_path: Path) -> None:
    result = runner.invoke(app, ["search", str(tmp_path / "missing.bin"), "x"])
    assert result.exit_code == 1
    assert "index file not found" in result.stderr
    assert "Traceback" not in result.output


def test_stats(index_file: Path) -> None:
    result = runner.invoke(app, ["stats", str(index_file), "--top", "3"])
    assert result.exit_code == 0, result.output
    assert "documents" in result.stdout
    assert "top 3 terms" in result.stdout


def test_vv_logs_debug_timings_to_stderr(index_file: Path) -> None:
    result = runner.invoke(app, ["-vv", "search", str(index_file), "gil", "--json"])
    assert result.exit_code == 0
    assert "Searcher.search took" in result.stderr
    assert all(line.startswith("{") for line in result.stdout.splitlines())
