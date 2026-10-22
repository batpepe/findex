"""One polite fetch: a timeout, bounded retries with backoff + jitter, ``Retry-After``.

Retried (at most ``max_retries`` times): timeouts, connection errors, ``429``
and ``5xx``. Not retried: any other ``4xx`` (a 404 will still be a 404 in a
second) and redirect loops. The delay before retry ``n`` is
``min(cap, base * 2**n) + uniform(0, 1)`` - exponential, so a struggling server
gets breathing room, plus jitter, so a hundred clients that failed together do
not all come back at the same instant. A ``Retry-After`` header on 429/503
replaces the computed delay when it is longer: the server said when.
"""

from __future__ import annotations

import asyncio
import email.utils
import logging
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import httpx

log = logging.getLogger(__name__)

USER_AGENT = "findex-crawler/0.6 (student search-engine project; +https://github.com/batpepe/findex)"
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
MAX_BODY = 5 * 2**20  # never keep more than 5 MiB of one page

type Sleep = Callable[[float], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class FetchResult:
    url: str
    final_url: str  # after redirects
    status: int  # 0 when no response arrived at all
    content_type: str = ""
    text: str = ""
    size: int = 0
    elapsed: float = 0.0
    attempts: int = 1
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.status == 200 and not self.error

    @property
    def is_html(self) -> bool:
        return "html" in self.content_type


def retry_after_seconds(value: str | None, now: float | None = None) -> float | None:
    """``Retry-After`` is either delta-seconds or an HTTP date."""
    if not value:
        return None
    value = value.strip()
    if value.isdigit():
        return float(value)
    try:
        when = email.utils.parsedate_to_datetime(value).timestamp()
    except TypeError, ValueError:
        return None
    return max(0.0, when - (time.time() if now is None else now))


def backoff_delay(attempt: int, base: float = 0.5, cap: float = 10.0, jitter: float = 1.0) -> float:
    return min(cap, base * 2**attempt) + random.uniform(0, jitter)


async def fetch(
    client: httpx.AsyncClient,
    url: str,
    *,
    timeout: float = 10.0,
    max_retries: int = 3,
    base_delay: float = 0.5,
    sleep: Sleep = asyncio.sleep,
) -> FetchResult:
    """GET ``url``; never raises for network or HTTP errors - they are in the result."""
    t0 = time.perf_counter()
    attempt = 0
    while True:
        error = ""
        delay: float | None = None
        try:
            async with asyncio.timeout(timeout):  # the whole request, body included
                response = await client.get(url, follow_redirects=True)
                body = response.content[:MAX_BODY]
        except TimeoutError:
            error = f"timeout after {timeout:.0f}s"
        except httpx.TooManyRedirects:
            return FetchResult(url, url, 0, elapsed=time.perf_counter() - t0, attempts=attempt + 1,
                               error="redirect loop")  # fmt: skip
        except httpx.TransportError as exc:  # connection refused/reset, DNS, TLS...
            error = f"{type(exc).__name__}: {exc}"
        else:
            status = response.status_code
            if status not in RETRY_STATUSES or attempt >= max_retries:
                content_type = response.headers.get("content-type", "")
                is_text = content_type.startswith("text/") or "html" in content_type
                text = body.decode(response.encoding or "utf-8", errors="replace") if is_text else ""
                return FetchResult(
                    url,
                    str(response.url),
                    status,
                    content_type,
                    text,
                    len(body),
                    time.perf_counter() - t0,
                    attempt + 1,
                    "" if status == 200 else f"HTTP {status}",
                )
            error = f"HTTP {status}"
            delay = retry_after_seconds(response.headers.get("retry-after")) if status in (429, 503) else None

        if attempt >= max_retries:
            return FetchResult(
                url, url, 0, elapsed=time.perf_counter() - t0, attempts=attempt + 1, error=error
            )
        wait = max(delay or 0.0, backoff_delay(attempt, base_delay))
        log.info("retry %d/%d for %s in %.1fs (%s)", attempt + 1, max_retries, url, wait, error)
        await sleep(wait)  # asyncio.sleep: the one sleep that does not block the loop
        attempt += 1
