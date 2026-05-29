"""Minimal Scrapy spider for the AI-agent-fingerprint baseline.

Crawls https://quotes.toscrape.com/, follows pagination, and extracts
quote text/author/tags. The interesting part for the fingerprint study is
self-identification: the spider explicitly logs the *outgoing* User-Agent
header that Scrapy attached to each request (the real wire value populated
by UserAgentMiddleware), not the documented default.

Run via run.sh (which wires up output/log paths and CLOSESPIDER limits) so
the crawl stays small and fast.
"""

import scrapy


class QuotesSpider(scrapy.Spider):
    name = "quotes"
    allowed_domains = ["quotes.toscrape.com", "httpbin.org"]
    start_urls = ["https://quotes.toscrape.com/"]

    # Keep defaults: ROBOTSTXT_OBEY = True (so robots.txt is fetched and
    # observable), default Protego robots parser, default Scrapy User-Agent.

    def parse(self, response):
        # Log the ACTUAL User-Agent Scrapy put on the wire for this request.
        # response.request.headers is populated by UserAgentMiddleware, so
        # this is the real sent value, not a doc claim.
        ua = response.request.headers.get(b"User-Agent")
        self.logger.info("UA sent (wire value): %r", ua)

        for quote in response.css("div.quote"):
            yield {
                "text": quote.css("span.text::text").get(),
                "author": quote.css("small.author::text").get(),
                "tags": quote.css("div.tags a.tag::text").getall(),
            }

        # Follow pagination (a couple of links) to prove link-following.
        next_page = response.css("li.next a::attr(href)").get()
        if next_page is not None:
            yield response.follow(next_page, callback=self.parse)

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
