# Scrapy baseline crawler

Baseline web crawler for the AI-agent-fingerprint study, using
[Scrapy](https://scrapy.org/) **2.14.1** (pinned via the Nix channel's
`python3Packages.scrapy`).

## Install / enter the environment

The host is NixOS; no `pip`/`scrapy` is on the global PATH. Everything runs
inside a `nix-shell`:

```sh
cd baselines/scrapy
nix-shell --run "scrapy version"      # -> Scrapy 2.14.1
```

`shell.nix` provides `python3Packages.scrapy`. No virtualenv, no `pip install`.

## Run the crawl

```sh
cd baselines/scrapy
nix-shell --run "./run.sh"
```

`run.sh` invokes `scrapy runspider quotes_spider.py` against
<https://quotes.toscrape.com/>, follows pagination, and stops early via
`CLOSESPIDER_PAGECOUNT=6` so the smoke test finishes in a few seconds. This is
a functional check, not a benchmark.

## Where output goes (all gitignored)

| Path                | Contents                                            |
| ------------------- | --------------------------------------------------- |
| `output/quotes.jl`  | Scraped items, one JSON object per line (`-O`)       |
| `logs/crawl.log`    | Full Scrapy DEBUG log (`LOG_FILE`): UA, robots.txt, fetched URLs |

Both `output/` and `logs/` match patterns in `baselines/.gitignore`.

## Observed default User-Agent

Scrapy does **not** print the outgoing UA in its standard log, so the spider
explicitly logs the wire value of `response.request.headers["User-Agent"]`
(populated by `UserAgentMiddleware`) and also performs one request to
`https://httpbin.org/headers` for a server-side reflection. Both agree:

```
Scrapy/2.14.1 (+https://scrapy.org)
```

- Wire log line: `[quotes] INFO: UA sent (wire value): b'Scrapy/2.14.1 (+https://scrapy.org)'`
- httpbin reflection: `"User-Agent": "Scrapy/2.14.1 (+https://scrapy.org)"`

The format is `Scrapy/<version> (+https://scrapy.org)`, so the version number
is leaked in the UA by default.

## robots.txt behavior

`ROBOTSTXT_OBEY` is left at the Scrapy default (`True`), so the crawler fetches
`/robots.txt` before crawling. quotes.toscrape.com returns **404** for
`/robots.txt`, which Scrapy's default Protego parser treats as allow-all, and
the crawl proceeds:

```
DEBUG: Crawled (404) <GET https://quotes.toscrape.com/robots.txt> (referer: None)
DEBUG: Crawled (200) <GET https://quotes.toscrape.com/>
DEBUG: Crawled (200) <GET https://quotes.toscrape.com/page/2/> (referer: https://quotes.toscrape.com/)
DEBUG: Crawled (200) <GET https://quotes.toscrape.com/page/3/> (referer: https://quotes.toscrape.com/page/2/)
```

## TLS stack (for the fingerprint study)

Scrapy networks on **Twisted** (asyncio reactor) with TLS provided by
**pyOpenSSL 26.0.0 over OpenSSL 3.6.2** (`cryptography` 48.0.0). The TLS
ClientHello / JA3-JA4 fingerprint therefore reflects Twisted + pyOpenSSL/OpenSSL,
distinct from a Go (`agent-scraper`) or JVM (Nutch/Heritrix) crawler.
