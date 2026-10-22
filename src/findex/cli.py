"""The ``findex`` command: index a corpus, search it, describe the index.

    findex [-v | -vv] index <corpus> --out index.bin [--positions] [--limit N]
    findex [-v | -vv] search <index> "<query>" [-k 10] [--scorer bm25|tfidf] [--json]
    findex [-v | -vv] stats <index> [--top 20]
    findex [-v | -vv] crawl <seed-url>... [--max-pages 500] [--concurrency 10] [--per-host 2] [--out FILE]

Results go to stdout (``--json`` emits one JSON object per line, for ``jq``);
progress, logs and errors go to stderr. Expected failures - a missing index, a
bad query - print one line and exit non-zero instead of a traceback.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Iterable, Iterator
from enum import StrEnum
from itertools import islice
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.live import Live
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TaskID, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text

from findex import __version__
from findex.corpus import Document, iter_documents
from findex.crawler.crawl import CrawlStats, url_log
from findex.crawler.crawl import crawl as crawl_pages
from findex.index import Index, build_index
from findex.parallel import DEFAULT_WORKERS, build_parallel, default_executor
from findex.query import QueryError
from findex.rank import SCORERS, Searcher
from findex.store import load, save

log = logging.getLogger("findex.cli")

app = typer.Typer(
    help="findex - full-text search over a corpus you choose.",
    no_args_is_help=True,
    add_completion=True,
    pretty_exceptions_show_locals=False,
)
out = Console()  # data
err = Console(stderr=True)  # everything else


class ExecutorChoice(StrEnum):
    serial = "serial"
    threads = "threads"
    processes = "processes"


class FormatName(StrEnum):
    binary = "binary"
    pickle = "pickle"


class ScorerName(StrEnum):
    bm25 = "bm25"
    tfidf = "tfidf"


def _version(value: bool) -> None:
    if value:
        print(f"findex {__version__}")
        raise typer.Exit


@app.callback()
def main(
    verbose: Annotated[
        int, typer.Option("--verbose", "-v", count=True, help="-v: info, -vv: debug (timings), to stderr")
    ] = 0,
    version: Annotated[
        bool, typer.Option("--version", callback=_version, is_eager=True, help="Show the version.")
    ] = False,
) -> None:
    """Configure logging once, here, for every subcommand (library code never does)."""
    level = (logging.WARNING, logging.INFO, logging.DEBUG)[min(verbose, 2)]
    logging.basicConfig(
        level=level,
        format="%(name)s: %(message)s",
        handlers=[RichHandler(console=err, show_path=False, show_time=False)],
        force=True,
    )


def _fail(message: str, code: int = 1) -> typer.Exit:
    err.print(f"[bold red]error:[/] {message}", highlight=False)
    return typer.Exit(code)


def _load(path: Path) -> Index:
    if not path.is_file():
        raise _fail(f"index file not found: {path} (build one with `findex index`)")
    try:
        return load(path)
    except (ValueError, TypeError, OSError) as exc:
        raise _fail(f"cannot load {path}: {exc}") from None


def _counting(docs: Iterable[Document], progress: Progress, task: TaskID) -> Iterator[Document]:
    for doc in docs:
        yield doc
        progress.advance(task)


@app.command()
def index(
    corpus: Annotated[Path, typer.Argument(help="Corpus directory (or a .jsonl file).")],
    out_path: Annotated[Path, typer.Option("--out", "-o", help="Where to write the index.")] = Path(
        "index.bin"
    ),
    positions: Annotated[bool, typer.Option(help="Store token positions (phrase queries).")] = True,
    limit: Annotated[
        int | None, typer.Option(help="Only the first N documents (streams serially).", min=1)
    ] = None,
    fmt: Annotated[
        FormatName, typer.Option("--format", help="binary (mmap, default) or pickle.")
    ] = FormatName.binary,
    executor: Annotated[
        ExecutorChoice | None,
        typer.Option(help="serial, threads or processes [default: processes; threads on a no-GIL build]"),
    ] = None,
    workers: Annotated[
        int, typer.Option(min=1, help="Worker threads/processes [default: CPU count, at most 8].")
    ] = DEFAULT_WORKERS,
) -> None:
    """Build an inverted index from a corpus and save it."""
    if not corpus.exists():
        raise _fail(f"corpus not found: {corpus}")
    chosen = executor.value if executor else default_executor()
    t0 = time.perf_counter()
    with Progress(
        SpinnerColumn(),
        TextColumn("indexing [bold]{task.completed:,}[/] documents"),
        TimeElapsedColumn(),
        console=err,
        transient=True,
    ) as progress:
        task = progress.add_task("index", total=None)
        if limit is not None:  # "the first N documents" needs the single ordered stream
            chosen = "serial"
            docs = _counting(islice(iter_documents(corpus), limit), progress, task)
            idx = build_index(docs, positions=positions)
        else:
            idx = build_parallel(
                corpus,
                executor=chosen,
                workers=workers,
                positions=positions,
                on_partial=lambda part: progress.advance(task, part.num_docs),
            )
    save(idx, out_path, fmt.value)
    size = out_path.stat().st_size / 2**20
    err.print(
        f"indexed {idx.num_docs:,} documents, {len(idx):,} terms in {time.perf_counter() - t0:.1f} s "
        f"({chosen}{'' if chosen == 'serial' else f', {workers} workers'}) "
        f"-> {out_path} ({size:.1f} MiB, {fmt.value})",
        highlight=False,
    )


@app.command()
def search(
    index_path: Annotated[Path, typer.Argument(metavar="INDEX", help="Index file.")],
    query: Annotated[str, typer.Argument(help='Query: words, OR, NOT, (...), "phrases".')],
    k: Annotated[int, typer.Option("--k", "-k", min=1, max=1000, help="How many results.")] = 10,
    scorer: Annotated[ScorerName, typer.Option(help="Ranking function.")] = ScorerName.bm25,
    as_json: Annotated[bool, typer.Option("--json", help="One JSON object per result line.")] = False,
) -> None:
    """Ranked search with highlighted snippets."""
    idx = _load(index_path)
    try:
        results = Searcher(idx).search(query, SCORERS[scorer.value], k)
    except QueryError as exc:
        raise _fail(f"bad query: {exc}", code=2) from None

    if as_json:
        for rank, r in enumerate(results, start=1):
            record = {
                "rank": rank,
                "doc_id": r.doc_id,
                "score": round(r.score, 4),
                "title": r.title,
                "path": r.path,
                "snippet": r.snippet.text if r.snippet else None,
                "highlights": r.snippet.highlights if r.snippet else [],
            }
            print(json.dumps(record, ensure_ascii=False))
        return

    table = Table(title=f"{results.total:,} matches for {query!r} ({scorer.value})", title_justify="left")
    table.add_column("#", justify="right", style="dim")
    table.add_column("score", justify="right")
    table.add_column("document")
    for rank, r in enumerate(results, start=1):
        cell = Text.assemble((r.title, "bold"), "  ", (r.path, "dim"))
        if r.snippet:
            cell.append("\n")
            for text, hit in r.snippet.fragments:
                cell.append(text, style="bold yellow" if hit else "")
        table.add_row(str(rank), f"{r.score:.3f}", cell)
    out.print(table)


@app.command()
def stats(
    index_path: Annotated[Path, typer.Argument(metavar="INDEX", help="Index file.")],
    top: Annotated[int, typer.Option(min=0, help="How many top terms to list.")] = 20,
) -> None:
    """Corpus and index statistics (the Lab 1 numbers, from the index)."""
    idx = _load(index_path)
    tokens = sum(idx.doc_lengths.values())
    longest = max(idx.doc_lengths, key=idx.doc_length)
    overview = Table(show_header=False, title=str(index_path), title_justify="left")
    overview.add_column(style="bold")
    overview.add_column(justify="right")
    for name, value in [
        ("documents", f"{idx.num_docs:,}"),
        ("tokens", f"{tokens:,}"),
        ("vocabulary", f"{len(idx):,}"),
        ("postings", f"{idx.num_postings():,}"),
        ("avg doc length", f"{idx.avg_doc_length:,.0f} tokens"),
        ("longest doc", f"{idx.doc_length(longest):,} tokens ({idx.title(longest)})"),
        ("positions", "yes" if idx.has_positions else "no"),
        ("file size", f"{index_path.stat().st_size / 2**20:.1f} MiB"),
    ]:
        overview.add_row(name, value)
    out.print(overview)
    if top:
        terms = Table(title=f"top {top} terms", title_justify="left")
        for column in ("#", "term", "occurrences", "documents"):
            terms.add_column(column, justify="left" if column == "term" else "right")
        for rank, term in enumerate(sorted(idx, key=idx.cf, reverse=True)[:top], start=1):
            terms.add_row(str(rank), term, f"{idx.cf(term):,}", f"{idx.df(term):,}")
        out.print(terms)


@app.command()
def crawl(
    seeds: Annotated[list[str], typer.Argument(help="Start URL(s).", metavar="SEED...")],
    out_path: Annotated[
        Path, typer.Option("--out", "-o", help="JSON-lines output (one page per line).")
    ] = Path("data/crawl/crawl.jsonl"),
    max_pages: Annotated[int, typer.Option(min=1, help="Stop after this many pages.")] = 500,
    concurrency: Annotated[int, typer.Option(min=1, max=200, help="Worker tasks / requests in flight.")] = 10,
    per_host: Annotated[int, typer.Option(min=1, help="Requests in flight per host.")] = 2,
    delay: Annotated[float, typer.Option(min=0.0, help="Seconds between request starts, per host.")] = 0.5,
    scope: Annotated[
        list[str] | None,
        typer.Option(help="URL prefix to stay under (repeatable) [default: each seed's dir]."),
    ] = None,
    log_path: Annotated[Path, typer.Option("--log", help="Per-URL log: status, bytes, time.")] = Path(
        "crawl.log"
    ),
    debug: Annotated[
        bool, typer.Option(help="asyncio debug mode: warn about anything blocking the loop.")
    ] = False,
) -> None:
    """Crawl politely (robots.txt, per-host limits, backoff) and stream pages to JSON lines."""
    stats = CrawlStats()
    t0 = time.perf_counter()
    coroutine = _crawl_to_jsonl(
        seeds, out_path, log_path, stats, max_pages, concurrency, per_host, delay, scope, debug
    )
    asyncio.run(coroutine, debug=debug)
    elapsed = time.perf_counter() - t0
    err.print(
        f"crawled {stats.pages:,} pages ({stats.fetched:,} fetched, {stats.errors:,} errors, "
        f"{stats.skipped_robots:,} disallowed by robots.txt) in {elapsed:.1f} s "
        f"({stats.pages / elapsed:.1f} pages/s) -> {out_path}; per-URL log: {log_path}",
        highlight=False,
    )
    if not stats.pages:
        raise _fail("no pages crawled (check the seed URL, the scope and crawl.log)")


def _crawl_dashboard(stats: CrawlStats) -> Text:
    return Text.assemble(
        ("crawling ", "bold"),
        f"{stats.pages:,} pages · {stats.pages_per_second:.1f} pages/s · ",
        f"in flight {stats.in_flight} · queued {stats.queued:,} · ",
        (f"errors {stats.errors}", "red" if stats.errors else ""),
        f" · robots-skipped {stats.skipped_robots} · {time.perf_counter() - stats.started:.0f}s",
    )


async def _crawl_to_jsonl(
    seeds: list[str],
    out_path: Path,
    log_path: Path,
    stats: CrawlStats,
    max_pages: int,
    concurrency: int,
    per_host: int,
    delay: float,
    scope: list[str] | None,
    debug: bool,
) -> None:
    if debug:
        asyncio.get_running_loop().slow_callback_duration = 0.05  # report any step blocking > 50 ms
    out_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    url_log.addHandler(handler)
    url_log.setLevel(logging.INFO)
    url_log.propagate = False
    try:
        with (
            out_path.open("w", encoding="utf-8") as f,
            Live(get_renderable=lambda: _crawl_dashboard(stats), console=err, refresh_per_second=4),
        ):
            pages = crawl_pages(
                seeds,
                max_pages=max_pages,
                concurrency=concurrency,
                per_host=per_host,
                delay=delay,
                scopes=scope,
                stats=stats,
            )
            async for page in pages:  # streaming: each page is written as soon as it arrives
                line = json.dumps(page.to_record(), ensure_ascii=False) + "\n"
                await asyncio.to_thread(f.write, line)  # file I/O stays off the event loop
    finally:
        url_log.removeHandler(handler)
        handler.close()

