"""robots.txt: fetched once per host, cached, consulted before every request."""

from __future__ import annotations

import asyncio
import logging
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

import httpx

from findex.crawler.fetch import USER_AGENT, fetch

log = logging.getLogger(__name__)

DISALLOW_ALL = ["User-agent: *", "Disallow: /"]


class RobotsCache:
    """``await robots.allowed(url)``; one robots.txt request per host, even under concurrency."""

    def __init__(self, client: httpx.AsyncClient, user_agent: str = USER_AGENT, max_retries: int = 1) -> None:
        self._client = client
        self._user_agent = user_agent
        self._max_retries = max_retries
        self._parsers: dict[str, asyncio.Task[RobotFileParser]] = {}

    async def _load(self, origin: str) -> RobotFileParser:
        parser = RobotFileParser(f"{origin}/robots.txt")
        result = await fetch(self._client, f"{origin}/robots.txt", max_retries=self._max_retries)
        if result.status == 200:
            lines = result.text.splitlines()
        elif result.status in (401, 403):
            lines = DISALLOW_ALL  # the convention: an access-controlled robots.txt means "keep out"
        elif 400 <= result.status < 500:
            lines = []  # no robots.txt: everything is allowed
        else:
            lines = DISALLOW_ALL  # 5xx or no answer: be conservative until we know
        parser.parse(lines)  # parse() also marks the parser as read; until then it allows nothing
        log.info("robots.txt for %s: HTTP %s", origin, result.status or result.error)
        return parser

    async def parser_for(self, url: str) -> RobotFileParser:
        parts = urlsplit(url)
        origin = f"{parts.scheme}://{parts.netloc}"
        task = self._parsers.get(origin)
        if task is None:  # the first coroutine to ask starts the fetch; the others await the same task
            task = self._parsers[origin] = asyncio.ensure_future(self._load(origin))
        return await asyncio.shield(task)

    async def allowed(self, url: str) -> bool:
        return (await self.parser_for(url)).can_fetch(self._user_agent, url)

    def close(self) -> None:
        """Cancel robots.txt fetches still in flight (the crawl is over)."""
        for task in self._parsers.values():
            task.cancel()

    async def crawl_delay(self, url: str) -> float | None:
        delay = (await self.parser_for(url)).crawl_delay(self._user_agent)
        return float(delay) if delay is not None else None
