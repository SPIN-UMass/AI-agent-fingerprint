# Traditional-crawler baselines

This directory holds three **traditional, HTTP-only web crawlers**, set up as a
fingerprinting **baseline** to contrast against the browser-driving AI agents
captured in [`../fingerprint_data/`](../fingerprint_data/) (browser_use,
skyvern, claude/gemini computer-use, autogen_websurfer).

Where the AI agents drive a *real headless Chrome* (near-real-browser TLS,
HTTP/2, full JS execution, `logger.js` interaction beacons), these traditional
crawlers are the opposite end of the spectrum: they **self-identify** in the
User-Agent, speak HTTP/1.1 with a **library-specific TLS stack** (so a stable,
distinctive JA3/JA4), and **execute no JavaScript** — so they emit *no*
`/collect` beacons at all. That gap is itself a strong, cheap signal.

The three picks are all canonical "traditional / archival research" crawlers
(see the survey in `~/Downloads/compass_artifact_*.md`, section A), chosen to
span three different TLS/HTTP-client stacks:

| Crawler | Lang | HTTP client | TLS / JA3 family | Default User-Agent (observed) | robots parser | JS | WARC |
|---|---|---|---|---|---|---|---|
| **Scrapy 2.14.1** | Python | Twisted | pyOpenSSL → **OpenSSL** | `Scrapy/2.14.1 (+https://scrapy.org)` | Protego | no | plugin |
| **Heritrix 3.15.0** | Java 17 | Apache HttpClient | **JVM JSSE** | `Mozilla/5.0 (compatible; heritrix/3.15.0 +<operator-url>)` | custom Java | no | **native** |
| **Apache Nutch 1.22** | Java | **OkHttp3** | JVM JSSE (via OkHttp) | `SPIN-baseline-nutch/Nutch-1.22-SNAPSHOT (https://github.com/SPIN-UMass/AI-agent-fingerprint)` | crawler-commons (RFC 9309) | no | plugin |

Heritrix and Nutch are both JVM/JSSE-backed but configure the TLS engine
differently (Apache HttpClient vs OkHttp3), so whether their JA3/JA4 hashes
actually diverge is an empirical question for the capture run — not assumed here.

## Reproducibility

Nothing is vendored into the repo. Each crawler is either a **nix-shell**
(Scrapy) or a **pinned Docker image** (Heritrix builds from the sha1-verified
official dist; Nutch uses the official image by tag). Re-running rebuilds/pulls
as needed.

| | install | run |
|---|---|---|
| Scrapy | `pkgs.python3Packages.scrapy` via `scrapy/shell.nix` | `cd scrapy && nix-shell --run ./run.sh` |
| Heritrix | `docker build` of `heritrix/Dockerfile` → `spin-heritrix:3.15.0` | `./heritrix/run.sh` |
| Nutch | `docker pull apache/nutch:release-1.22` | `cd nutch && ./run.sh` |

Each crawler's own `README.md` documents its UA, robots behavior, TLS stack,
and the exact run sequence (Heritrix in particular is server-driven via its
REST API — `run.sh` does create → build → launch → unpause → terminate).

## Test target

The smoke tests crawl **<https://quotes.toscrape.com/>** — a small Zyte
scraping sandbox with pagination (so link-following is exercised) that finishes
in seconds. It is a **functional check** that each crawler installs, runs,
follows links, and self-identifies; it is *not* the fingerprint-capture run.
(`quotes.toscrape.com/robots.txt` returns 404 = allow-all; use
`books.toscrape.com` if you want a target that serves a real `robots.txt`.)

### Pointing them at the fingerprint server

For an actual capture run, repoint each crawler at the fingerprint testbed
(`uxbehaviorsuite.com`, or a local instance) and, for comparability with the
agent traces, restrict each to the same page set the agents hit
(`index.html`, `subscribe-v1/2/3.html`, `s2`–`s5`):

- **Scrapy** — change `start_urls` in `scrapy/quotes_spider.py` (drop the
  `httpbin` UA cross-check) and `allowed_domains`.
- **Heritrix** — change the `<seed>` and `metadata.operatorContactUrl` in
  `heritrix/crawler-beans.cxml`.
- **Nutch** — change `nutch/urls/seed.txt` and the host rule in
  `nutch/regex-urlfilter.txt`.

Then split the captured `requests-*.jsonl` by time window with
`fingerprint_data/split_traces.py` (same methodology used for the agents),
into a *separate* baseline directory — don't mix into `agent_transcripts/`.

## Artifacts / cleanliness

`baselines/.gitignore` keeps the repo to **configs + run scripts + READMEs
only**. All crawl output (scraped items, WARCs, Nutch crawldb/segments, logs,
Heritrix jobs) is routed to gitignored `output/` / `logs/` dirs and is safe to
delete. The two Docker images (`spin-heritrix:3.15.0` ≈ 768 MB,
`apache/nutch:release-1.22` ≈ 2.15 GB) live in Docker's image store, not the
repo; `docker rmi` them to reclaim disk.
