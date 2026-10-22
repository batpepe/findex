"""findex on the web: a JSON API and a small HTML UI over one FastAPI app.

    INDEX_PATH=index.bin uvicorn findex.web.app:app          # or: findex serve index.bin

API: ``GET /search``, ``GET /docs/{doc_id}``, ``GET /stats``, ``GET /health``
(OpenAPI browser at ``/docs``, the schema at ``/openapi.json``).
UI: ``GET /`` (search page) and ``GET /doc/{doc_id}`` (one document).

The index is loaded once, in the ``lifespan`` context manager, and reaches the
routes through ``Depends(get_searcher)`` - which tests replace with a tiny
in-memory index via ``app.dependency_overrides``. ``/search`` is a plain
``def``: ranking is CPU work, so FastAPI runs it in its thread pool instead of
on the event loop (``SEARCH_ASYNC=true`` registers it as ``async def``, for the
benchmark that justifies this choice).
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
import time
import uuid
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from findex import __version__
from findex.query import QueryError, parse, positive_terms
from findex.rank import SCORERS, Searcher, SearchResult
from findex.snippets import highlight_html
from findex.store import load
from findex.web.models import (
    DocumentResponse,
    ErrorResponse,
    HealthResponse,
    SearchHit,
    SearchResponse,
    StatsResponse,
    source_url,
)
from findex.web.settings import Settings, get_settings

log = logging.getLogger("findex.web")
HERE = Path(__file__).parent
MAX_DEPTH = 1000  # page * k: deeper pagination is refused rather than ranked
templates = Jinja2Templates(directory=HERE / "templates")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": round(record.created, 3),
            "level": record.levelname.lower(),
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            entry["exc"] = self.formatException(record.exc_info)
        return json.dumps(entry, ensure_ascii=False)


def configure_logging(settings: Settings) -> None:
    handler = logging.StreamHandler(sys.stdout)  # 12-factor rule XI: logs are a stream on stdout
    handler.setFormatter(
        JsonFormatter() if settings.log_json else logging.Formatter("%(levelname)s %(name)s: %(message)s")
    )
    root = logging.getLogger("findex")
    root.handlers[:] = [handler]
    root.setLevel(settings.log_level.upper())
    root.propagate = False


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    settings = get_settings()
    configure_logging(settings)
    app.state.started = time.time()
    app.state.searcher = None
    try:
        index = load(settings.index_path)  # once per process, not per request
    except (OSError, ValueError, TypeError) as exc:
        log.error("index not loaded from %s: %s - /health reports 503", settings.index_path, exc)
        index = None
    if index is not None:
        app.state.searcher = Searcher(index, cache_size=settings.search_cache_size)
        log.info("loaded %r from %s", index, settings.index_path)
    try:
        yield
    finally:  # graceful shutdown (uvicorn handles SIGTERM, then this runs)
        if index is not None:
            index.close()


def get_searcher(request: Request) -> Searcher:
    searcher: Searcher | None = getattr(request.app.state, "searcher", None)
    if searcher is None:
        raise HTTPException(status_code=503, detail="index not loaded")
    return searcher


def get_optional_searcher(request: Request) -> Searcher | None:
    return getattr(request.app.state, "searcher", None)


# Plain aliases, not PEP 695 `type` statements: FastAPI reads these annotations at
# import time and does not unwrap a TypeAliasType (typer has the same limitation).
SearcherDep = Annotated[Searcher, Depends(get_searcher)]
ScorerName = Literal["bm25", "tfidf"]
QueryText = Annotated[str, Query(min_length=1, max_length=200, description="The query")]
PerPage = Annotated[int, Query(ge=1, le=100, description="Results per page")]
PageNumber = Annotated[int, Query(ge=1, le=100, description="1-based page number")]


@dataclass(frozen=True, slots=True)
class Page:
    """One page of ranked results, before it becomes JSON or HTML."""

    results: list[SearchResult]
    total: int
    took_ms: float


def ranked_page(searcher: Searcher, q: str, k: int, page: int, scorer: ScorerName) -> Page:
    if page * k > MAX_DEPTH:
        raise HTTPException(status_code=422, detail=f"page * k must be at most {MAX_DEPTH}")
    t0 = time.perf_counter()
    try:
        results = searcher.search(q, SCORERS[scorer], k=page * k)
    except QueryError as exc:
        raise HTTPException(status_code=400, detail=f"bad query: {exc}") from None
    took = round((time.perf_counter() - t0) * 1000, 2)
    return Page(list(results[(page - 1) * k : page * k]), results.total, took)


def run_search(searcher: Searcher, q: str, k: int, page: int, scorer: ScorerName) -> SearchResponse:
    result = ranked_page(searcher, q, k, page, scorer)
    return SearchResponse(
        query=q,
        scorer=scorer,
        total=result.total,
        page=page,
        k=k,
        pages=max(1, math.ceil(result.total / k)),
        took_ms=result.took_ms,
        results=[SearchHit.from_result(r) for r in result.results],
    )


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="findex",
        version=__version__,
        description="Full-text search over the Python documentation, every PEP and the PyPA/devguide/"
        "typing sites. BM25 or TF-IDF ranking; query language: words (AND), `OR`, `NOT`, "
        '`(...)`, `"phrases"`.',
        lifespan=lifespan,
    )
    app.state.started = time.time()  # the lifespan resets it; set here too for apps used without one
    if settings.cors_origins:
        app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["GET"])
    app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")

    @app.middleware("http")
    async def request_context(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        request.state.request_id = request_id
        t0 = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            log.exception("[%s] unhandled error on %s %s", request_id, request.method, request.url.path)
            body = ErrorResponse(detail="internal server error", request_id=request_id)
            response = JSONResponse(status_code=500, content=body.model_dump())
        response.headers["x-request-id"] = request_id
        log.info(
            "[%s] %s %s -> %d (%.1f ms)",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            (time.perf_counter() - t0) * 1000,
        )
        return response

    responses: dict[int | str, dict[str, object]] = {
        400: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    }

    if os.environ.get("SEARCH_ASYNC", "").lower() in ("1", "true", "yes"):

        @app.get("/search", response_model=SearchResponse, responses=responses, tags=["api"])
        async def search_async(  # pyright: ignore[reportUnusedFunction] - registered by the decorator
            searcher: SearcherDep,
            q: QueryText,
            k: PerPage = 10,
            page: PageNumber = 1,
            scorer: ScorerName = "bm25",
        ) -> SearchResponse:
            """Ranked search - as `async def`: the CPU work blocks the event loop (benchmark only)."""
            return run_search(searcher, q, k, page, scorer)

    else:

        @app.get("/search", response_model=SearchResponse, responses=responses, tags=["api"])
        def search(  # pyright: ignore[reportUnusedFunction]
            searcher: SearcherDep,
            q: QueryText,
            k: PerPage = 10,
            page: PageNumber = 1,
            scorer: ScorerName = "bm25",
        ) -> SearchResponse:
            """Ranked search with highlighted snippets and pagination."""
            return run_search(searcher, q, k, page, scorer)

    @app.get(
        "/docs/{doc_id}",
        response_model=DocumentResponse,
        responses={404: {"model": ErrorResponse}},
        tags=["api"],
    )
    def document(doc_id: int, searcher: SearcherDep) -> DocumentResponse:  # pyright: ignore[reportUnusedFunction]
        """One document, with its full text."""
        if doc_id not in searcher.index.doc_meta:
            raise HTTPException(status_code=404, detail=f"no document {doc_id}")
        return DocumentResponse.from_index(searcher.index, doc_id)

    @app.get("/stats", response_model=StatsResponse, tags=["api"])
    def stats(  # pyright: ignore[reportUnusedFunction]
        request: Request, searcher: SearcherDep, settings: Annotated[Settings, Depends(get_settings)]
    ) -> StatsResponse:
        """Index size and service uptime."""
        index = searcher.index
        cache = searcher.cache_info()
        size = settings.index_path.stat().st_size if settings.index_path.is_file() else 0
        return StatsResponse(
            version=__version__,
            documents=index.num_docs,
            terms=len(index),
            tokens=sum(index.doc_lengths.values()),
            avg_doc_length=round(index.avg_doc_length, 1),
            positions=index.has_positions,
            index_size_bytes=size,
            uptime_s=round(time.time() - request.app.state.started, 1),
            query_cache_hits=cache.hits,
            query_cache_misses=cache.misses,
        )

    @app.get(
        "/health", response_model=HealthResponse, responses={503: {"model": HealthResponse}}, tags=["ops"]
    )
    def health(  # pyright: ignore[reportUnusedFunction]
        searcher: Annotated[Searcher | None, Depends(get_optional_searcher)],
    ) -> HealthResponse | JSONResponse:
        """Liveness/readiness: 200 once the index is loaded, 503 before (or if it never was)."""
        if searcher is None:
            body = HealthResponse(status="unavailable", documents=0)
            return JSONResponse(status_code=503, content=body.model_dump())
        return HealthResponse(status="ok", documents=searcher.index.num_docs)

    # --- the HTML UI: the same app, routes that return HTML -------------------------------

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def home(  # pyright: ignore[reportUnusedFunction]
        request: Request,
        searcher: SearcherDep,
        q: Annotated[str, Query(max_length=200)] = "",
        page: Annotated[int, Query(ge=1, le=100)] = 1,
        scorer: ScorerName = "bm25",
    ) -> HTMLResponse:
        context: dict[str, object] = {
            "q": q,
            "scorer": scorer,
            "index": searcher.index,
            "version": __version__,
            "page": page,
            "source_url": source_url,
        }
        status = 200
        if q.strip():
            try:
                result = ranked_page(searcher, q, 10, page, scorer)
                context["result"] = result
                context["pages"] = max(1, math.ceil(result.total / 10))
            except HTTPException as exc:
                context["error"] = exc.detail
                status = exc.status_code
        return templates.TemplateResponse(request, "index.html", context, status_code=status)

    @app.get("/doc/{doc_id}", response_class=HTMLResponse, include_in_schema=False)
    def doc_page(  # pyright: ignore[reportUnusedFunction]
        request: Request, doc_id: int, searcher: SearcherDep, q: Annotated[str, Query(max_length=200)] = ""
    ) -> HTMLResponse:
        index = searcher.index
        if doc_id not in index.doc_meta:
            raise HTTPException(status_code=404, detail=f"no document {doc_id}")
        terms: frozenset[str] = frozenset()
        if q.strip():
            with suppress(QueryError):
                terms = frozenset(positive_terms(parse(q)))
        doc = DocumentResponse.from_index(index, doc_id)
        context = {"doc": doc, "q": q, "body": highlight_html(doc.text, terms), "version": __version__}
        return templates.TemplateResponse(request, "doc.html", context)

    return app


app = create_app()
__all__ = ["app", "create_app", "get_searcher", "source_url"]
