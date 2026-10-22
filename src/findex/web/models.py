"""Pydantic models at the HTTP boundary.

Inside, ``findex`` keeps its dataclasses (``Posting``, ``SearchResult``...):
fast, no validation cost. Pydantic validates and documents what crosses the
wire - the seam is the ``from_result`` / ``from_index`` converters below.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from findex.index import Index
from findex.rank import SearchResult


def source_url(path: str) -> str | None:
    """Crawled documents keep their URL as the path; files have none."""
    return path if path.startswith(("http://", "https://")) else None


class SearchHit(BaseModel):
    doc_id: int
    title: str
    path: str
    url: str | None = Field(description="Original web page, for crawled documents")
    score: float
    snippet: str = Field(description="Best-matching window of the text, plain")
    highlights: list[str] = Field(description="The words in the snippet that matched the query")

    @classmethod
    def from_result(cls, r: SearchResult) -> SearchHit:
        return cls(
            doc_id=r.doc_id,
            title=r.title,
            path=r.path,
            url=source_url(r.path),
            score=round(r.score, 4),
            snippet=r.snippet.text if r.snippet else "",
            highlights=r.snippet.highlights if r.snippet else [],
        )


class SearchResponse(BaseModel):
    query: str
    scorer: str
    total: int = Field(description="Documents matching the query (all pages)")
    page: int
    k: int = Field(description="Results per page")
    pages: int
    took_ms: float
    results: list[SearchHit]


EXCERPT = 600  # characters shown of a crawled page: we index other sites, we do not republish them


class DocumentResponse(BaseModel):
    doc_id: int
    title: str
    path: str
    url: str | None
    length: int = Field(description="Tokens in the document")
    text: str = Field(description="Full text; for crawled web pages only an excerpt (see `truncated`)")
    truncated: bool = Field(description="True for crawled pages: read the rest at `url`")

    @classmethod
    def from_index(cls, index: Index, doc_id: int) -> DocumentResponse:
        meta = index.doc_meta[doc_id]
        url = source_url(meta.path)
        text = index.text(doc_id)
        truncated = url is not None and len(text) > EXCERPT
        return cls(
            doc_id=doc_id,
            title=meta.title,
            path=meta.path,
            url=url,
            length=index.doc_length(doc_id),
            text=text[:EXCERPT].rstrip() + " …" if truncated else text,
            truncated=truncated,
        )


class StatsResponse(BaseModel):
    version: str
    documents: int
    terms: int
    tokens: int
    avg_doc_length: float
    positions: bool
    index_size_bytes: int
    uptime_s: float
    query_cache_hits: int
    query_cache_misses: int


class HealthResponse(BaseModel):
    status: str
    documents: int


class ErrorResponse(BaseModel):
    detail: str
    request_id: str | None = None
