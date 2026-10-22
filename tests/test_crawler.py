"""The async crawler, tested without the network (httpx.MockTransport) (Lab 6)."""

import asyncio
import email.utils
import json
import threading
import time
from collections import Counter
from collections.abc import Callable, Coroutine, Iterator
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from socketserver import TCPServer

import httpx
import pytest
from typer.testing import CliRunner

from findex.cli import app
from findex.crawler import RobotsCache, canonicalize, crawl, fetch
from findex.crawler.crawl import CrawlStats, Page, extract
from findex.crawler.fetch import retry_after_seconds

type Handler = (
    Callable[[httpx.Request], httpx.Response]
    | Callable[[httpx.Request], Coroutine[None, None, httpx.Response]]
)


def html(body: str, title: str = "T") -> httpx.Response:
    return httpx.Response(200, headers={"content-type": "text/html"}, text=f"<title>{title}</title>{body}")


def client_for(handler: Handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler), max_redirects=5)


class SleepSpy:
    def __init__(self) -> None:
        self.calls: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)


def run_fetch(handler: Handler, **kwargs: float) -> tuple[object, SleepSpy]:
    spy = SleepSpy()

    async def go() -> object:
        async with client_for(handler) as client:
            return await fetch(client, "https://site.test/page", sleep=spy, **kwargs)  # type: ignore[arg-type]

    return asyncio.run(go()), spy


# --- fetch ------------------------------------------------------------------


def test_fetch_ok() -> None:
    result, spy = run_fetch(lambda r: html("<p>hello</p>"))
    assert result.ok and result.is_html and result.attempts == 1  # type: ignore[attr-defined]
    assert spy.calls == []


def test_fetch_retries_5xx_with_exponential_backoff() -> None:
    answers = iter([httpx.Response(503), httpx.Response(502), html("ok")])
    result, spy = run_fetch(lambda r: next(answers))
    assert result.ok and result.attempts == 3  # type: ignore[attr-defined]
    assert 0.5 <= spy.calls[0] <= 1.5 and 1.0 <= spy.calls[1] <= 2.0  # base * 2**n + jitter in [0, 1)


def test_fetch_honours_retry_after_on_429() -> None:
    answers = iter([httpx.Response(429, headers={"retry-after": "7"}), html("ok")])
    result, spy = run_fetch(lambda r: next(answers))
    assert result.ok  # type: ignore[attr-defined]
    assert spy.calls == [7.0]  # the server's 7 s beats our computed ~1 s


def test_fetch_does_not_retry_404() -> None:
    result, spy = run_fetch(lambda r: httpx.Response(404))
    assert (result.status, result.attempts, result.error) == (404, 1, "HTTP 404")  # type: ignore[attr-defined]
    assert spy.calls == []


def test_fetch_gives_up_after_max_retries_of_timeouts() -> None:
    async def slow(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(1)
        return html("too late")

    result, spy = run_fetch(slow, timeout=0.02)
    assert result.status == 0 and result.attempts == 4  # type: ignore[attr-defined]
    assert "timeout" in result.error  # type: ignore[attr-defined]
    assert len(spy.calls) == 3


def test_fetch_retries_connection_errors() -> None:
    attempts = Counter[str]()

    def flaky(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise httpx.ConnectError("connection reset", request=request)
        return html("ok")

    result, _ = run_fetch(flaky)
    assert result.ok and result.attempts == 2  # type: ignore[attr-defined]


def test_fetch_reports_a_redirect_loop_without_retrying() -> None:
    def loop(request: httpx.Request) -> httpx.Response:
        target = "/b" if request.url.path == "/page" else "/page"
        return httpx.Response(302, headers={"location": target})

    result, spy = run_fetch(loop)
    assert (result.status, result.error) == (0, "redirect loop")  # type: ignore[attr-defined]
    assert spy.calls == []


def test_retry_after_parsing() -> None:
    assert retry_after_seconds("120") == 120
    now = time.time()
    assert retry_after_seconds(email.utils.formatdate(now + 30, usegmt=True), now=now) == pytest.approx(
        30, abs=1
    )
    assert retry_after_seconds("soon") is None
    assert retry_after_seconds(None) is None


# --- robots.txt ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("robots", "path", "allowed"),
    [
        (httpx.Response(200, text="User-agent: *\nDisallow: /private/\n"), "/docs/a.html", True),
        (httpx.Response(200, text="User-agent: *\nDisallow: /private/\n"), "/private/x", False),
        (httpx.Response(404), "/anything", True),
        (httpx.Response(403), "/anything", False),
        (httpx.Response(500), "/anything", False),
    ],
)
def test_robots_rules(robots: httpx.Response, path: str, allowed: bool) -> None:
    async def go() -> bool:
        async with client_for(lambda r: robots) as client:
            return await RobotsCache(client, max_retries=0).allowed(f"https://site.test{path}")

    assert asyncio.run(go()) is allowed


def test_robots_is_fetched_once_per_host_under_concurrency() -> None:
    requests = Counter[str]()

    async def handler(request: httpx.Request) -> httpx.Response:
        requests[str(request.url)] += 1
        await asyncio.sleep(0.01)  # slow enough for all 20 callers to arrive first
        return httpx.Response(200, text="User-agent: *\nCrawl-delay: 2\n")

    async def go() -> list[bool]:
        async with client_for(handler) as client:
            robots = RobotsCache(client)
            answers = await asyncio.gather(*(robots.allowed(f"https://site.test/{i}") for i in range(20)))
            assert await robots.crawl_delay("https://site.test/") == 2.0
            return answers

    assert all(asyncio.run(go()))
    assert requests == {"https://site.test/robots.txt": 1}


# --- URLs and HTML ------------------------------------------------------------


@pytest.mark.parametrize(
    ("url", "canonical"),
    [
        ("HTTPS://Docs.Example.ORG/a/b.html#section", "https://docs.example.org/a/b.html"),
        ("https://example.org", "https://example.org/"),
        ("https://example.org:443/x/", "https://example.org/x/"),
        ("http://example.org:8080/x", "http://example.org:8080/x"),
        ("https://example.org/a/./b/../c/index.html", "https://example.org/a/c/"),
        ("https://example.org/s?b=2&a=1", "https://example.org/s?a=1&b=2"),
        ("mailto:someone@example.org", None),
        ("javascript:void(0)", None),
    ],
)
def test_canonicalize(url: str, canonical: str | None) -> None:
    assert canonicalize(url) == canonical


def test_extract_title_text_and_links() -> None:
    page = """<html><head><title>Guide</title><script>var x = 1;</script></head><body>
      <nav><a href="/nav-only">Nav</a></nav>
      <main><h1>Packaging</h1><p>Build a <a href="wheel.html#top">wheel</a>.</p></main>
      <footer>(c) footer</footer></body></html>"""
    title, text, links = extract(page, "https://site.test/docs/index.html")
    assert title == "Guide"
    assert "Packaging" in text and "wheel" in text
    assert "var x" not in text and "footer" not in text
    assert links == ["https://site.test/nav-only", "https://site.test/docs/wheel.html"]


# --- the whole crawl ----------------------------------------------------------

SITE = {
    "/robots.txt": "User-agent: *\nDisallow: /docs/private/\n",
    "/docs/": '<a href="a.html">a</a> <a href="b.html#frag">b</a> <a href="./">self</a> '
    '<a href="../outside.html">out of scope</a> <a href="https://other.test/x">other host</a> '
    '<a href="private/secret.html">disallowed</a> <a href="manual.pdf">pdf</a> <a href="gone.html">404</a>',
    "/docs/a.html": '<a href="b.html">b again</a> <a href="index.html">home</a> <p>Alpha page</p>',
    "/docs/b.html": "<p>Beta page</p>",
}


def site_handler(log: list[str]) -> Handler:
    def handler(request: httpx.Request) -> httpx.Response:
        log.append(str(request.url))
        if request.url.host != "site.test":
            raise AssertionError(f"left the allowed domain: {request.url}")
        body = SITE.get(request.url.path)
        if body is None:
            return httpx.Response(404)
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text=body)
        return html(body, title=request.url.path)

    return handler


def crawl_all(max_pages: int = 50, concurrency: int = 4) -> tuple[list[Page], list[str], CrawlStats]:
    requests: list[str] = []
    stats = CrawlStats()

    async def go() -> list[Page]:
        async with client_for(site_handler(requests)) as client:
            pages = crawl(
                ["https://site.test/docs/"],
                max_pages=max_pages,
                concurrency=concurrency,
                delay=0,
                client=client,
                stats=stats,
            )
            return [page async for page in pages]

    return asyncio.run(go(), debug=True), requests, stats


def test_crawl_fetches_every_in_scope_page_exactly_once() -> None:
    pages, requests, stats = crawl_all()
    assert sorted(p.url for p in pages) == [
        "https://site.test/docs/",
        "https://site.test/docs/a.html",
        "https://site.test/docs/b.html",
    ]
    fetched = [r for r in requests if not r.endswith("robots.txt")]
    assert len(fetched) == len(set(fetched)) == 4  # 3 pages + the 404, no duplicates
    assert not any("private" in r or "outside" in r or r.endswith(".pdf") for r in requests)
    assert (stats.pages, stats.errors, stats.skipped_robots) == (3, 1, 1)
    assert {p.title for p in pages} == {"/docs/", "/docs/a.html", "/docs/b.html"}


def test_crawl_stops_cleanly_at_max_pages() -> None:
    pages, _, stats = crawl_all(max_pages=2, concurrency=1)
    assert len(pages) == stats.pages == 2


# --- the CLI against a real (local) HTTP server -------------------------------


class QuickHTTPServer(ThreadingHTTPServer):
    def server_bind(self) -> None:  # HTTPServer.server_bind does a reverse-DNS getfqdn(): 30 s on some Macs
        TCPServer.server_bind(self)
        self.server_name, self.server_port = "127.0.0.1", self.server_address[1]


@pytest.fixture
def local_site(tmp_path: Path) -> Iterator[str]:
    (tmp_path / "robots.txt").write_text("User-agent: *\nAllow: /\n")
    (tmp_path / "index.html").write_text(
        '<title>Home</title><main>Welcome to the walrus docs. <a href="guide.html">guide</a></main>'
    )
    (tmp_path / "guide.html").write_text("<title>Guide</title><main>The walrus operator guide.</main>")
    handler = partial(SimpleHTTPRequestHandler, directory=str(tmp_path))
    SimpleHTTPRequestHandler.log_message = lambda *args: None  # type: ignore[method-assign]  # keep pytest quiet
    server = QuickHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}/"
    server.shutdown()


def test_cli_crawl_then_index_then_search(local_site: str, tmp_path: Path) -> None:
    runner = CliRunner()
    out, log = tmp_path / "crawl" / "site.jsonl", tmp_path / "crawl.log"
    result = runner.invoke(app, ["crawl", local_site, "--out", str(out), "--log", str(log), "--delay", "0"])
    assert result.exit_code == 0, result.output
    records = [json.loads(line) for line in out.read_text().splitlines()]
    assert sorted(r["title"] for r in records) == ["Guide", "Home"]
    assert "200" in log.read_text()

    index_path = tmp_path / "i.bin"
    assert runner.invoke(app, ["index", str(out.parent), "--out", str(index_path)]).exit_code == 0
    result = runner.invoke(app, ["search", str(index_path), "walrus operator", "--json"])
    assert json.loads(result.stdout.splitlines()[0])["path"] == local_site + "guide.html"


# --- the politeness frontier ----------------------------------------------------


def test_host_frontier_serves_the_ready_host_first() -> None:
    from findex.crawler.crawl import HostFrontier

    async def go() -> list[tuple[str, float]]:
        frontier = HostFrontier(delay=0.2)
        for url in ["https://a.test/1", "https://a.test/2", "https://a.test/3", "https://b.test/1"]:
            frontier.put_nowait(url)
        t0 = time.monotonic()
        handed_out: list[tuple[str, float]] = []
        for _ in range(4):
            url = await frontier.get()
            handed_out.append((url, time.monotonic() - t0))
            frontier.task_done()
        await frontier.join()
        return handed_out

    order = asyncio.run(go())
    # a FIFO queue would make b.test wait behind a.test's cool-down; here it goes second
    assert [url for url, _ in order] == [
        "https://a.test/1",
        "https://b.test/1",
        "https://a.test/2",
        "https://a.test/3",
    ]
    assert order[1][1] < 0.05  # b.test right away
    assert 0.15 < order[2][1] < 0.35 and 0.35 < order[3][1] < 0.6  # a.test 0.2 s apart


def test_host_frontier_shutdown_wakes_waiters() -> None:
    from findex.crawler.crawl import HostFrontier

    async def go() -> None:
        frontier = HostFrontier(delay=0)
        waiter = asyncio.create_task(frontier.get())
        await asyncio.sleep(0.01)
        frontier.put_nowait("https://a.test/1")
        assert await waiter == "https://a.test/1"
        frontier.put_nowait("https://a.test/2")
        frontier.shutdown(immediate=True)
        with pytest.raises(asyncio.QueueShutDown):
            await frontier.get()
        frontier.task_done()  # the one URL that was handed out
        await asyncio.wait_for(frontier.join(), timeout=1)

    asyncio.run(go())
