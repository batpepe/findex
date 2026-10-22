"""The ``python -m findex.<module>`` entry points from Labs 1-3 still work."""

from pathlib import Path

import pytest

from findex import index as index_module
from findex import search as search_module
from findex import stats as stats_module
from findex.corpus import iter_documents
from findex.stats import corpus_stats, corpus_stats_eager


def test_lazy_and_eager_stats_agree(corpus_dir: Path) -> None:
    lazy = corpus_stats(iter_documents(corpus_dir))
    eager = corpus_stats_eager(corpus_dir)
    assert (lazy.documents, lazy.tokens, lazy.vocabulary) == (eager.documents, eager.tokens, eager.vocabulary)
    assert lazy.top_terms(3) == eager.top_terms(3)
    assert lazy.top_terms(1) == [("the", 5)]


def test_stats_main(corpus_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    stats_module.main([str(corpus_dir), "--limit", "3", "--top", "2"])
    out = capsys.readouterr().out
    assert "documents:   3" in out
    assert "top 2 terms" in out


def test_index_and_search_mains(corpus_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "i.pkl"
    index_module.main([str(corpus_dir), "--out", str(path), "--format", "pickle", "--positions"])
    assert "documents:   6" in capsys.readouterr().out
    search_module.main([str(path), "gil", "--boolean", "--engine", "set"])
    assert "2 matching documents for 'gil' (set engine)" in capsys.readouterr().out
    search_module.main([str(path), '"event loop"', "--explain", "-k", "1"])
    out = capsys.readouterr().out
    assert 'parsed as: "event loop"' in out
    assert "tf=" in out


def test_search_main_rejects_bad_queries(index_file: Path) -> None:
    with pytest.raises(SystemExit) as exit_info:
        search_module.main([str(index_file), "NOT"])
    assert exit_info.value.code == 2


@pytest.mark.slow
def test_real_corpus_smoke() -> None:
    """Build the full index from data/ (skipped when the corpus is not downloaded, e.g. in CI)."""
    data = Path(__file__).parents[1] / "data"
    if not data.is_dir():
        pytest.skip("corpus not downloaded (./scripts/get-corpus.sh)")
    from findex.index import build_index
    from findex.rank import Searcher

    index = build_index(iter_documents(data), positions=True)
    assert index.num_docs >= 1_297
    top = Searcher(index).search("structural pattern matching", k=5, snippets=False)
    assert any(r.path == "peps/pep-0634.rst" for r in top)
