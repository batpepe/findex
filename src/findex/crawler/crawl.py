"""The crawl: a queue-based frontier, worker tasks, per-host politeness.

Shape (the shape of most real async programs)::

    seeds -> [frontier: per-host queues] -> N workers (TaskGroup) -> [pages: asyncio.Queue] -> async for page

* ``concurrency`` worker tasks pull a URL, ask ``robots.txt``, wait for their
  host's turn, fetch, extract the text and links, and push new in-scope URLs.
* Politeness: a global semaphore bounds requests in flight; a per-host
  semaphore bounds requests to one server; the ``HostFrontier`` spaces out
  each host's request *starts* by a minimum delay (``robots.txt``'s
  Crawl-delay wins if it is longer) and always serves the host that is ready
  first, so no worker idles behind a host that is cooling down.
* Every URL is canonicalized before the ``seen`` check, so ``/a/``, ``/a/#x``
  and ``/a/index.html`` are fetched once.
* ``crawl()`` is an async generator - but it never ``yield``s inside the
  ``TaskGroup``: the group lives in a separate task and hands pages over through
  a queue. Yielding from inside a task group or a timeout scope lets a
  consumer's cancellation land in the wrong place (PEP 789, which this corpus
  contains).
* HTML parsing is CPU work: it runs in a thread (``asyncio.to_thread``) so the
  loop keeps serving sockets. Indexing is not done here at all.
"""

from __future__ import annotations

import asyncio
import contextlib
import heapq
import itertools
import logging
import posixpath
import time
from collections import deque
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit

import httpx
from selectolax.lexbor import LexborHTMLParser

from findex.crawler.fetch import USER_AGENT, FetchResult, fetch
from findex.crawler.robots import RobotsCache

log = logging.getLogger(__name__)
url_log = logging.getLogger("findex.crawl.urls")  # one line per URL; the CLI sends it to crawl.log

SKIP_SUFFIXES = (
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".css", ".js", ".pdf", ".zip", ".gz", ".bz2",
    ".tar", ".whl", ".txt", ".xml", ".json", ".woff", ".woff2", ".ttf", ".mp4",
)  # fmt: skip
SKIP_PARTS = ("/_sources/", "/_static/", "/_images/", "/_downloads/", "/genindex", "/search.html")


@dataclass(frozen=True, slots=True)
class Page:
    url: str
    title: str
    text: str
    fetched_at: str

    def to_record(self) -> dict[str, str]:
        return {"url": self.url, "title": self.title, "text": self.text, "fetched_at": self.fetched_at}


@dataclass(slots=True)
class CrawlStats:
    """Live counters for the progress display and the benchmark."""

    started: float = field(default_factory=time.perf_counter)
    pages: int = 0
    fetched: int = 0
    errors: int = 0
    skipped_robots: int = 0
    in_flight: int = 0
    queued: int = 0

    @property
    def pages_per_second(self) -> float:
        return self.pages / max(1e-9, time.perf_counter() - self.started)


def canonicalize(url: str, base: str | None = None) -> str | None:
    """One spelling per resource: lower-case scheme/host, no fragment, no default port,
    sorted query, ``index.html`` and duplicate slashes folded. ``None`` if not http(s)."""
    if base is not None:
        url = urljoin(base, url)
    parts = urlsplit(url.strip())
    scheme = parts.scheme.lower()
    if scheme not in ("http", "https") or not parts.hostname:
        return None
    host = parts.hostname.lower()
    if parts.port and parts.port != {"http": 80, "https": 443}[scheme]:
        host = f"{host}:{parts.port}"
    path = parts.path or "/"
    keep_slash = path.endswith("/")
    path = posixpath.normpath(path)
    path = "/" if path == "." else path
    if path.endswith("/index.html"):
        path, keep_slash = path.removesuffix("index.html"), True
    if keep_slash and not path.endswith("/"):
        path += "/"
    query = urlencode(sorted(parse_qsl(parts.query, keep_blank_values=True)))
    return urlunsplit((scheme, host, path, query, ""))


def extract(html: str, url: str) -> tuple[str, str, list[str]]:
    """``(title, main text, absolute links)`` of one HTML page (CPU work: run it in a thread)."""
    tree = LexborHTMLParser(html)
    title_node = tree.css_first("title")
    title = title_node.text(strip=True) if title_node else url
    links = [href for a in tree.css("a[href]") if (href := a.attributes.get("href"))]
    for junk in tree.css("script, style, nav, header, footer, aside, .headerlink, .sphinxsidebar"):
        junk.decompose()
    main = tree.css_first("main, [role=main], article, .body") or tree.body
    text = main.text(separator="\n", strip=True) if main else ""
    return title, text, [link for link in (canonicalize(href, url) for href in links) if link]


def in_scope(url: str, scopes: Iterable[str]) -> bool:
    path = urlsplit(url).path
    return (
        any(url.startswith(scope) for scope in scopes)
        and not path.endswith(SKIP_SUFFIXES)
        and not any(part in path for part in SKIP_PARTS)
    )


class HostFrontier:
    """The frontier, in politeness order rather than FIFO order.

    One FIFO per host, plus a heap of ``(time this host may be contacted again,
    host)``. ``get()`` hands out a URL from whichever host becomes ready first and
    pushes that host's next turn ``delay`` seconds into the future. With a plain
    FIFO queue, workers took URLs of one host in a row and all waited on its
    delay while other hosts sat idle - head-of-line blocking, measured in the
    README. Same interface as ``asyncio.Queue``: ``put_nowait``, ``get``,
    ``task_done``, ``join``, ``qsize``, ``shutdown``.
    """

    def __init__(self, delay: float) -> None:
        self.default_delay = delay
        self._delays: dict[str, float] = {}
        self._queues: dict[str, deque[str]] = {}
        self._heap: list[tuple[float, int, str]] = []
        self._scheduled: set[str] = set()
        self._next_ok: dict[str, float] = {}
        self._order = itertools.count()  # tie-breaker: hosts ready at the same time keep FIFO order
        self._changed = asyncio.Event()
        self._finished = asyncio.Event()
        self._finished.set()
        self._unfinished = 0
        self._closed = False

    def set_delay(self, host: str, delay: float) -> None:
        self._delays[host] = delay

    def put_nowait(self, url: str) -> None:
        host = urlsplit(url).netloc
        self._queues.setdefault(host, deque()).append(url)
        self._unfinished += 1
        self._finished.clear()
        if host not in self._scheduled:
            self._schedule(host)
        self._changed.set()

    def _schedule(self, host: str) -> None:
        heapq.heappush(self._heap, (self._next_ok.get(host, 0.0), next(self._order), host))
        self._scheduled.add(host)

    async def get(self) -> str:
        while True:
            if self._closed:
                raise asyncio.QueueShutDown
            if not self._heap:
                self._changed.clear()  # no await since the check: no wake-up can be lost
                await self._changed.wait()
                continue
            ready_at, _, host = self._heap[0]
            now = time.monotonic()
            if ready_at > now:  # the soonest host is still cooling down: wait for it, or for news
                self._changed.clear()
                with contextlib.suppress(TimeoutError):
                    async with asyncio.timeout(ready_at - now):
                        await self._changed.wait()
                continue
            heapq.heappop(self._heap)
            self._scheduled.discard(host)
            url = self._queues[host].popleft()
            self._next_ok[host] = now + self._delays.get(host, self.default_delay)
            if self._queues[host]:
                self._schedule(host)
            return url

    def task_done(self) -> None:
        self._unfinished -= 1
        if self._unfinished <= 0:
            self._finished.set()

    async def join(self) -> None:
        await self._finished.wait()

    def qsize(self) -> int:
        return sum(map(len, self._queues.values()))

    def shutdown(self, immediate: bool = False) -> None:
        self._closed = True
        if immediate:  # drop what was never handed out; URLs being visited still call task_done()
            self._unfinished -= self.qsize()
            self._queues.clear()
            self._heap.clear()
            if self._unfinished <= 0:
                self._finished.set()
        self._changed.set()


@dataclass(slots=True)
class _Crawl:
    client: httpx.AsyncClient
    scopes: list[str]
    max_pages: int
    concurrency: int
    per_host: int
    delay: float
    stats: CrawlStats
    out: asyncio.Queue[Page | None]
    frontier: HostFrontier = field(init=False)
    seen: set[str] = field(default_factory=set[str])
    in_flight: dict[str, asyncio.Semaphore] = field(default_factory=dict[str, asyncio.Semaphore])
    robots: RobotsCache | None = None
    global_limit: asyncio.Semaphore | None = None
    done: asyncio.Event = field(default_factory=asyncio.Event)

    def __post_init__(self) -> None:
        self.frontier = HostFrontier(self.delay)

    def enqueue(self, url: str) -> None:
        if url in self.seen or not in_scope(url, self.scopes):
            return
        self.seen.add(url)
        self.frontier.put_nowait(url)
        self.stats.queued = self.frontier.qsize()

    async def host_limit(self, url: str) -> asyncio.Semaphore:
        """The host's in-flight cap; on first contact, adopt robots.txt's Crawl-delay if longer."""
        host = urlsplit(url).netloc
        if host not in self.in_flight:
            assert self.robots is not None
            crawl_delay = await self.robots.crawl_delay(url) or 0.0
            if crawl_delay > self.delay:
                self.frontier.set_delay(host, crawl_delay)
            self.in_flight.setdefault(host, asyncio.Semaphore(self.per_host))
        return self.in_flight[host]

    async def worker(self) -> None:
        while True:
            try:
                url = await self.frontier.get()
            except asyncio.QueueShutDown:
                return
            try:
                await self.visit(url)
            finally:
                self.frontier.task_done()
                self.stats.queued = self.frontier.qsize()

    async def visit(self, url: str) -> None:
        assert self.robots is not None and self.global_limit is not None
        if not await self.robots.allowed(url):
            self.stats.skipped_robots += 1
            url_log.info("robots   -        -        - %s", url)
            return
        host_limit = await self.host_limit(url)
        async with self.global_limit, host_limit:  # the frontier already spaced this host's turns
            self.stats.in_flight += 1
            try:
                result = await fetch(self.client, url)
            finally:
                self.stats.in_flight -= 1
        self.stats.fetched += 1
        self._log(result)
        if not result.ok or not result.is_html:
            self.stats.errors += not result.ok
            return
        title, text, links = await asyncio.to_thread(extract, result.text, result.final_url)
        final = canonicalize(result.final_url) or url
        self.seen.add(final)  # a redirect target is a URL we now have, too
        for link in links:
            self.enqueue(link)
        if text and self.stats.pages < self.max_pages:
            self.stats.pages += 1
            await self.out.put(Page(final, title, text, datetime.now(UTC).isoformat(timespec="seconds")))
            if self.stats.pages >= self.max_pages:
                self.done.set()

    @staticmethod
    def _log(r: FetchResult) -> None:
        url_log.info(
            "%-8s %8d %6.0fms %d %s%s",
            r.status or "ERR",
            r.size,
            r.elapsed * 1000,
            r.attempts,
            r.url,
            f"  [{r.error}]" if r.error else (f"  -> {r.final_url}" if r.final_url != r.url else ""),
        )

    async def run(self, seeds: list[str]) -> None:
        self.robots = RobotsCache(self.client)
        self.global_limit = asyncio.Semaphore(self.concurrency)
        for seed in seeds:
            self.enqueue(seed)
        try:
            async with asyncio.TaskGroup() as tg:
                workers = [tg.create_task(self.worker()) for _ in range(self.concurrency)]
                exhausted = asyncio.ensure_future(self.frontier.join())
                limit_hit = asyncio.ensure_future(self.done.wait())
                await asyncio.wait([exhausted, limit_hit], return_when=asyncio.FIRST_COMPLETED)
                exhausted.cancel()
                limit_hit.cancel()
                self.frontier.shutdown(immediate=True)  # idle workers get QueueShutDown and return
                for task in workers:  # busy ones are cancelled at their current await
                    task.cancel()
        finally:
            self.robots.close()
            await self.out.put(None)  # tell the consumer we are done (or failed)


async def crawl(
    seeds: Iterable[str],
    *,
    max_pages: int = 500,
    concurrency: int = 10,
    per_host: int = 2,
    delay: float = 0.5,
    scopes: Iterable[str] | None = None,
    client: httpx.AsyncClient | None = None,
    stats: CrawlStats | None = None,
) -> AsyncIterator[Page]:
    """Yield pages as they are fetched. ``scopes`` are URL prefixes to stay under
    (default: each seed's own directory), which is also the domain allowlist."""
    seed_urls = [url for url in (canonicalize(s) for s in seeds) if url]
    scope_list = list(scopes) if scopes is not None else [s.rsplit("/", 1)[0] + "/" for s in seed_urls]
    owns_client = client is None
    client = client or httpx.AsyncClient(
        headers={"User-Agent": USER_AGENT},
        timeout=httpx.Timeout(10.0),
        limits=httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency),
        max_redirects=5,
    )
    pages: asyncio.Queue[Page | None] = asyncio.Queue(maxsize=max(1, concurrency))
    machine = _Crawl(
        client, scope_list, max_pages, concurrency, per_host, delay, stats or CrawlStats(), pages
    )
    runner = asyncio.create_task(machine.run(seed_urls))
    try:
        while (page := await pages.get()) is not None:
            yield page
        await runner  # re-raise anything that went wrong in the workers
    finally:
        if not runner.done():  # the consumer stopped early: stop the workers
            runner.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await runner
        if owns_client:
            await client.aclose()
