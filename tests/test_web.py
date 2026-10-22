"""The web API and UI through FastAPI's TestClient, with a tiny injected index (Lab 7)."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from findex.index import Index
from findex.rank import Searcher
from findex.web.app import app, get_optional_searcher, get_searcher
from findex.web.settings import get_settings


@pytest.fixture
def client(corpus_index: Index) -> Iterator[TestClient]:
    """No lifespan, no index file: the dependency is overridden with the fixture index."""
    searcher = Searcher(corpus_index)
    app.dependency_overrides[get_searcher] = lambda: searcher
    app.dependency_overrides[get_optional_searcher] = lambda: searcher
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_search(client: TestClient) -> None:
    response = client.get("/search", params={"q": "event loop"})
    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 2 and body["page"] == 1 and body["pages"] == 1
    assert {r["title"] for r in body["results"]} == {"Coroutines and tasks", "Crawled A"}
    crawled = next(r for r in body["results"] if r["title"] == "Crawled A")
    assert crawled["url"] == "https://example.org/a"
    assert set(crawled["highlights"]) == {"event", "loop"}
    assert response.headers["x-request-id"]


def test_search_pagination(client: TestClient) -> None:
    first = client.get("/search", params={"q": "event loop", "k": 1}).json()
    second = client.get("/search", params={"q": "event loop", "k": 1, "page": 2}).json()
    assert first["pages"] == second["pages"] == 2
    assert first["total"] == second["total"] == 2
    assert first["results"][0]["doc_id"] != second["results"][0]["doc_id"]


@pytest.mark.parametrize(
    "params",
    [
        {},  # q is required
        {"q": ""},
        {"q": "x" * 201},
        {"q": "gil", "k": "abc"},
        {"q": "gil", "k": 0},
        {"q": "gil", "k": 101},
        {"q": "gil", "scorer": "pagerank"},
        {"q": "gil", "k": 100, "page": 11},  # deeper than 1000 results
    ],
)
def test_search_validation_is_422(client: TestClient, params: dict[str, object]) -> None:
    response = client.get("/search", params=params)  # type: ignore[arg-type]
    assert response.status_code == 422
    assert "detail" in response.json()


def test_bad_query_syntax_is_400(client: TestClient) -> None:
    response = client.get("/search", params={"q": "(event OR"})
    assert response.status_code == 400
    assert response.json()["detail"] == "bad query: query ends where a term was expected"


def test_document(client: TestClient) -> None:
    body = client.get("/docs/5").json()
    assert body["title"] == "PEP 9999 \N{EN DASH} The Walrus Guide"
    assert "walrus operator" in body["text"]
    assert body["url"] is None


def test_crawled_pages_are_shown_as_excerpts(client: TestClient, corpus_index: Index) -> None:
    crawled = client.get("/docs/0").json()  # https://example.org/a, a short page
    assert crawled["url"] == "https://example.org/a" and crawled["truncated"] is False
    from findex.web import models

    long_text = "word " * 500
    assert len(long_text) > models.EXCERPT
    original = corpus_index.text
    corpus_index.text = lambda doc_id: long_text  # type: ignore[method-assign]  # a long crawled page
    try:
        body = client.get("/docs/0").json()
    finally:
        corpus_index.text = original  # type: ignore[method-assign]
    assert body["truncated"] is True and len(body["text"]) <= models.EXCERPT + 2


def test_unknown_document_is_404_and_bad_id_is_422(client: TestClient) -> None:
    missing = client.get("/docs/999")
    assert missing.status_code == 404
    assert missing.json() == {"detail": "no document 999"}
    assert client.get("/docs/abc").status_code == 422


def test_stats(client: TestClient) -> None:
    body = client.get("/stats").json()
    assert body["documents"] == 6
    assert body["positions"] is True
    assert body["terms"] > 0


def test_health(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok", "documents": 6}


def test_health_is_503_without_an_index() -> None:
    response = TestClient(app).get("/health")  # no override, lifespan never ran: no index
    assert response.status_code == 503
    assert response.json()["status"] == "unavailable"
    assert TestClient(app).get("/search", params={"q": "x"}).status_code == 503


def test_lifespan_loads_the_index_from_the_environment(
    index_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    get_settings.cache_clear()
    monkeypatch.setenv("INDEX_PATH", str(index_file))
    with TestClient(app) as client:  # entering the context runs the lifespan
        assert client.get("/health").json() == {"status": "ok", "documents": 6}
        assert client.get("/search", params={"q": "walrus"}).json()["total"] == 1
    monkeypatch.setenv("INDEX_PATH", str(tmp_path / "missing.bin"))
    get_settings.cache_clear()
    with TestClient(app) as client:
        assert client.get("/health").status_code == 503
    get_settings.cache_clear()


def test_unexpected_errors_are_a_clean_500_with_a_request_id(client: TestClient) -> None:
    def broken() -> Searcher:
        raise RuntimeError("boom")

    app.dependency_overrides[get_searcher] = broken
    response = client.get("/search", params={"q": "gil"}, headers={"x-request-id": "req-42"})
    assert response.status_code == 500
    assert response.json() == {"detail": "internal server error", "request_id": "req-42"}
    assert response.headers["x-request-id"] == "req-42"


def test_openapi_documents_every_route(client: TestClient) -> None:
    paths = client.get("/openapi.json").json()["paths"]
    assert {"/search", "/docs/{doc_id}", "/stats", "/health"} <= set(paths)
    assert client.get("/docs").status_code == 200  # the interactive API browser


def test_ui_pages(client: TestClient) -> None:
    home = client.get("/")
    assert home.status_code == 200 and "<form" in home.text
    results = client.get("/", params={"q": "walrus"})
    assert "The Walrus Guide" in results.text and "<mark>walrus</mark>" in results.text.lower()
    doc = client.get("/doc/5", params={"q": "walrus"})
    assert doc.status_code == 200 and "<mark>walrus</mark>" in doc.text
    bad = client.get("/", params={"q": "(walrus"})
    assert bad.status_code == 400 and "bad query" in bad.text
    assert client.get("/static/style.css").status_code == 200
