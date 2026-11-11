"""Semantic and hybrid search (Lab 8, M3), offline: the hashing embedder and a tiny ONNX model."""

import math
import shutil
from collections.abc import Generator, Iterator
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from findex.cli import app as cli
from findex.corpus import Document
from findex.index import Index, build_index
from findex.rank import Searcher, SemanticUnavailable
from findex.semantic import (
    HashingEmbedder,
    MiniLM,
    VectorIndex,
    build_vectors,
    chunk_spans,
    load_vectors,
    mean_pool,
    normalize_rows,
    rrf,
    save_vectors,
    semantic_rank,
    semantic_text,
    vector_paths,
)
from findex.web.app import app, get_optional_searcher, get_searcher

TINY_MODEL = Path(__file__).parent / "data" / "tiny-minilm"


@pytest.fixture(scope="module")
def vectors(corpus_index: Index) -> VectorIndex:
    return build_vectors(corpus_index, HashingEmbedder(), chunk_words=8, stride=6)


@pytest.fixture(scope="module")
def searcher(corpus_index: Index, vectors: VectorIndex) -> Searcher:
    return Searcher(corpus_index, vectors=vectors, embedder=HashingEmbedder())


def test_chunks_overlap_and_cover_the_whole_text() -> None:
    text = " ".join(f"w{i}" for i in range(25))
    spans = chunk_spans(text, chunk_words=10, stride=8)
    pieces = [text[a:b].split() for a, b in spans]
    assert [p[0] for p in pieces] == ["w0", "w8", "w16"]  # a new chunk every 8 words
    assert pieces[0][-2:] == pieces[1][:2]  # 10 - 8 = 2 words of overlap
    assert pieces[-1][-1] == "w24"  # nothing at the end is lost
    assert chunk_spans("") == []
    assert len(chunk_spans("just a few words")) == 1


def test_rows_are_unit_vectors_so_dot_product_is_cosine() -> None:
    rows = normalize_rows(np.array([[3.0, 4.0], [0.0, 0.0], [1.0, 1.0]]))
    assert np.allclose(np.linalg.norm(rows[[0, 2]], axis=1), 1.0)
    assert rows[1].tolist() == [0.0, 0.0]  # no division by zero
    assert math.isclose(float(rows[0] @ rows[2]), (3 + 4) / (5 * math.sqrt(2)), rel_tol=1e-6)


def test_mean_pooling_ignores_padding() -> None:
    hidden = np.array([[[1.0, 0.0], [3.0, 0.0], [100.0, 100.0]]], dtype=np.float32)  # last token = [PAD]
    pooled = mean_pool(hidden, np.array([[1, 1, 0]]))
    assert np.allclose(pooled, [[1.0, 0.0]])  # mean of the two real tokens (2, 0), normalized


def test_rrf_rewards_agreement_and_needs_no_score_scales() -> None:
    fused = dict(rrf([1, 2, 3], [3, 1, 4]))
    assert fused[1] == pytest.approx(1 / 61 + 1 / 62)
    assert fused[3] == pytest.approx(1 / 63 + 1 / 61)
    assert fused[4] == pytest.approx(1 / 63)  # only in one list
    assert [doc for doc, _ in rrf([1, 2, 3], [3, 1, 4])][:2] == [1, 3]  # in both lists, high in both
    assert rrf([7], [8]) == [(7, 1 / 61), (8, 1 / 61)]  # a tie: lower doc id first


def test_operators_are_not_embedded() -> None:
    assert semantic_text('"event loop" NOT (asyncio OR tkinter)') == "event loop asyncio tkinter"


def test_vector_index_groups_chunks_by_document(corpus_index: Index, vectors: VectorIndex) -> None:
    assert vectors.dim == 256 and len(vectors) >= corpus_index.num_docs
    assert vectors.doc_ids.tolist() == sorted(corpus_index.doc_meta)  # every document has chunks
    for position, doc_id in enumerate(vectors.doc_ids.tolist()):
        assert set(vectors.chunks[vectors.segment(position), 0].tolist()) == {doc_id}


def test_best_chunk_decides_the_document_score(corpus_index: Index, vectors: VectorIndex) -> None:
    query = HashingEmbedder().embed(["walrus operator"])[0]
    (total, top), best = semantic_rank(vectors, query, k=3)
    assert total == corpus_index.num_docs
    pep = next(d for d, m in corpus_index.doc_meta.items() if m.path.endswith("pep-9999.rst"))
    assert top[0][0] == pep
    sims = vectors.vectors @ query
    assert top[0][1] == pytest.approx(float(sims[vectors.chunks[:, 0] == pep].max()))
    assert int(vectors.chunks[best[pep], 0]) == pep


def test_save_and_load_vectors_memory_mapped(
    corpus_index: Index, vectors: VectorIndex, tmp_path: Path
) -> None:
    index_path = tmp_path / "index.bin"
    save_vectors(vectors, index_path, chunk_words=8)
    assert all(p.is_file() for p in vector_paths(index_path))
    loaded = load_vectors(index_path)
    assert loaded is not None and loaded.model == "hashing"
    assert isinstance(loaded.vectors, np.memmap)  # not read into memory: shared via the page cache
    assert np.array_equal(loaded.vectors, vectors.vectors) and np.array_equal(loaded.chunks, vectors.chunks)
    assert load_vectors(tmp_path / "other.bin") is None


def test_searcher_modes(searcher: Searcher) -> None:
    keyword = searcher.search("walrus", mode="keyword")
    semantic = searcher.search("walrus operator", mode="semantic", k=3)
    hybrid = searcher.search("walrus", mode="hybrid", k=3)
    assert keyword[0].title == semantic[0].title == hybrid[0].title == "PEP 9999 \N{EN DASH} The Walrus Guide"
    assert hybrid[0].score == pytest.approx(2 / 61)  # first in both rankings
    assert semantic[0].snippet is not None and "walrus" in semantic[0].snippet.text.lower()


def test_semantic_free_text_need_not_be_a_valid_query(searcher: Searcher) -> None:
    assert searcher.search('what does "NOT do (', mode="semantic", k=1)  # no QueryError


def test_semantic_without_embeddings_is_refused(corpus_index: Index) -> None:
    with pytest.raises(SemanticUnavailable):
        Searcher(corpus_index).search("x", mode="semantic")


# --- the web API and the CLI ------------------------------------------------------------


@pytest.fixture
def client(searcher: Searcher) -> Iterator[TestClient]:
    app.dependency_overrides[get_searcher] = lambda: searcher
    app.dependency_overrides[get_optional_searcher] = lambda: searcher
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.mark.parametrize("mode", ["keyword", "semantic", "hybrid"])
def test_api_modes(client: TestClient, mode: str) -> None:
    response = client.get("/search", params={"q": "walrus", "mode": mode})
    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == mode
    assert body["results"][0]["path"].endswith("pep-9999.rst")


def test_api_rejects_unknown_mode_and_reports_health(client: TestClient) -> None:
    assert client.get("/search", params={"q": "x", "mode": "vibes"}).status_code == 422
    assert client.get("/health").json()["semantic"] is True
    page = client.get("/", params={"q": "walrus", "mode": "hybrid"})
    assert page.status_code == 200 and 'name="mode"' in page.text and "fused (RRF)" in page.text


def test_api_semantic_without_embeddings_is_400(corpus_index: Index) -> None:
    plain = Searcher(corpus_index)
    app.dependency_overrides[get_searcher] = lambda: plain
    try:
        response = TestClient(app).get("/search", params={"q": "x", "mode": "semantic"})
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 400
    assert "no embeddings" in response.json()["detail"]


def test_cli_embed_then_search_by_meaning(index_file: Path, tmp_path: Path) -> None:
    index_path = tmp_path / "index.bin"
    shutil.copy(index_file, index_path)
    runner = CliRunner()
    result = runner.invoke(
        cli, ["embed", str(index_path), "--model", "hashing", "--chunk-words", "16", "--stride", "12"]
    )
    assert result.exit_code == 0, result.output
    assert "chunks" in result.stderr
    found = runner.invoke(cli, ["search", str(index_path), "walrus operator", "--mode", "hybrid", "--json"])
    assert found.exit_code == 0
    assert '"path": "peps/pep-9999.rst"' in found.stdout.splitlines()[0]
    missing = runner.invoke(cli, ["search", str(index_file), "x", "--mode", "semantic"])
    assert missing.exit_code == 1 and "findex embed" in missing.stderr


def test_lifespan_loads_embeddings_next_to_the_index(
    index_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from findex.web.settings import get_settings

    index_path = tmp_path / "index.bin"
    shutil.copy(index_file, index_path)
    assert CliRunner().invoke(cli, ["embed", str(index_path), "--model", "hashing"]).exit_code == 0
    get_settings.cache_clear()
    monkeypatch.setenv("INDEX_PATH", str(index_path))
    try:
        with TestClient(app) as client:
            assert client.get("/health").json()["semantic"] is True
            assert client.get("/search", params={"q": "walrus", "mode": "semantic"}).status_code == 200
    finally:
        get_settings.cache_clear()


# --- the real model's code path, on a tiny ONNX model with the same interface -----------


def tiny_minilm(tmp_path: Path) -> MiniLM:
    pytest.importorskip("onnxruntime")
    pytest.importorskip("tokenizers")
    shutil.copytree(TINY_MODEL, tmp_path / MiniLM.name)
    return MiniLM.load(tmp_path, download=False)


def test_minilm_pipeline_runs_onnx_and_mean_pools(tmp_path: Path) -> None:
    model = tiny_minilm(tmp_path)
    assert model.dim == 8  # read from the model's output shape (384 for the real one)
    texts = ["the cats", "walrus", "the event loop python gil", "unknownword"]
    vectors = model.embed(texts)
    assert vectors.shape == (4, 8) and vectors.dtype == np.float32
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-6)
    alone = np.concatenate([model.embed([t]) for t in texts])  # batched with padding == one by one
    assert np.allclose(vectors, alone, atol=1e-6)


def test_minilm_refuses_to_download_when_told_not_to(tmp_path: Path) -> None:
    pytest.importorskip("onnxruntime")
    with pytest.raises(FileNotFoundError):
        MiniLM.load(tmp_path, download=False)


def test_semantic_search_with_the_onnx_pipeline(tmp_path: Path) -> None:
    model = tiny_minilm(tmp_path)
    docs = [
        Document(i, f"d{i}", f"d{i}", text) for i, text in enumerate(["the walrus", "event loop", "cats"])
    ]
    index = build_index(docs, positions=True)
    searcher = Searcher(index, vectors=build_vectors(index, model), embedder=model)
    assert searcher.search("walrus", mode="semantic", k=1)[0].doc_id == 0


def test_model_download_is_pinned_and_checked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A tampered or changed model.onnx is refused, not cached (no network: httpx.stream is faked)."""
    import contextlib

    import httpx

    from findex.semantic import MODEL_REVISION, model_files

    urls: list[str] = []

    class FakeResponse:
        def raise_for_status(self) -> None: ...

        def iter_bytes(self, size: int) -> Iterator[bytes]:
            yield b"not the real model"

    @contextlib.contextmanager
    def fake_stream(method: str, url: str, **kwargs: object) -> Generator[FakeResponse]:
        urls.append(url)
        yield FakeResponse()

    monkeypatch.setattr(httpx, "stream", fake_stream)
    with pytest.raises(OSError, match="SHA-256"):
        model_files(MiniLM.name, tmp_path)
    assert all(f"/resolve/{MODEL_REVISION}/" in url for url in urls)  # never a moving "main"
    assert not (tmp_path / MiniLM.name / "model.onnx").exists()  # nothing half-verified is kept
