"""Minimal Scrapy spider for the AI-agent-fingerprint baseline.

By default crawls https://quotes.toscrape.com/ (a public scraping sandbox),
follows links, and extracts quote text/author/tags. Set the TARGET_URL env
var to point the SAME spider at another site (e.g. the fingerprint capture
server); it then just follows same-domain <a href> links so it traverses
whatever pages the target links to.

The interesting part for the fingerprint study is self-identification: the
spider explicitly logs the *outgoing* User-Agent header that Scrapy attached
to each request (the real wire value populated by UserAgentMiddleware), not
the documented default.

Run via run.sh (which wires up output/log paths and CLOSESPIDER limits) so
the crawl stays small and fast.
"""

import os
from urllib.parse import urlparse

import scrapy

# Target is configurable so this baseline can run against either the public
# quotes sandbox (default) or the study's fingerprint server.
TARGET_URL = os.environ.get("TARGET_URL", "https://quotes.toscrape.com/")
_TARGET_HOST = urlparse(TARGET_URL).netloc


class QuotesSpider(scrapy.Spider):
    name = "quotes"
    # Stay on the target host. httpbin is allowed only for the optional
    # quotes-sandbox cross-check below.
    allowed_domains = [_TARGET_HOST, "httpbin.org"]
    start_urls = [TARGET_URL]

    # Keep defaults: ROBOTSTXT_OBEY = True (so robots.txt is fetched and
    # observable), default Protego robots parser, default Scrapy User-Agent.

    def parse(self, response):
        # Log the ACTUAL User-Agent Scrapy put on the wire for this request.
        # response.request.headers is populated by UserAgentMiddleware, so
        # this is the real sent value, not a doc claim.
        ua = response.request.headers.get(b"User-Agent")
        self.logger.info("UA sent (wire value): %r", ua)

        # Opportunistic quotes extraction -- a no-op on non-quotes targets.
        for quote in response.css("div.quote"):
            yield {
                "text": quote.css("span.text::text").get(),
                "author": quote.css("small.author::text").get(),
                "tags": quote.css("div.tags a.tag::text").getall(),
            }

        # Follow same-domain navigational links. Scrapy's dupefilter prevents
        # revisits and allowed_domains keeps us on-site; CLOSESPIDER_PAGECOUNT
        # (set in run.sh) bounds the crawl. Works for any target.
        for href in response.css("a::attr(href)").getall():
            yield response.follow(href, callback=self.parse)

        # Belt-and-suspenders: hit httpbin once for a server-side reflection
        # of the UA. Optional -- if httpbin is flaky the crawl still succeeds
        # because response.request.headers above is already solid evidence.
        if response.url.rstrip("/") == "https://quotes.toscrape.com":
            yield scrapy.Request(
                "https://httpbin.org/headers",
                callback=self.parse_httpbin,
                errback=self.httpbin_failed,
                dont_filter=True,
            )

    def parse_httpbin(self, response):
        self.logger.info("httpbin server-side headers: %s", response.text)

    def httpbin_failed(self, failure):
        self.logger.warning("httpbin check failed (non-fatal): %s", failure.value)
