"""An async, polite web crawler that feeds the corpus (Lab 6)."""

from findex.crawler.crawl import CrawlStats, Page, canonicalize, crawl
from findex.crawler.fetch import USER_AGENT, FetchResult, fetch
from findex.crawler.robots import RobotsCache

__all__ = ["USER_AGENT", "CrawlStats", "FetchResult", "Page", "RobotsCache", "canonicalize", "crawl", "fetch"]
