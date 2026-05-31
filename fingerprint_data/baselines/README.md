# Baseline crawler fingerprint captures

Server-side fingerprints captured from the three traditional-crawler baselines
(set up under [`baselines/`](../../baselines/)) crawling the live fingerprint
server. These are the deterministic, non-LLM counterparts to the agent traces
in [`../splitted_traces/`](../splitted_traces/): they let us contrast what a
classic web/archival crawler looks like on the wire against the AI browser
agents.

## Layout

```
fingerprint_data/baselines/
├── scrapy/    { requests.jsonl, capture.json }
├── heritrix/  { requests.jsonl, capture.json }
└── nutch/     { requests.jsonl, capture.json }
```

- `requests.jsonl` — the server's per-request fingerprint records (same schema
  as `../logs/requests-*.jsonl`: `tls` with ja3/ja4, `http2` akamai fingerprint,
  `http` with ordered headers), one JSON object per line, verbatim from the
  server log.
- `capture.json` — provenance + a computed summary (UTC windows, observed UA,
  TLS hashes, protocol, header orderings, paths) for that crawler.

## How these were captured

- **Target:** `https://uxbehaviorsuite.com/` (= the EC2 capture server,
  `13.220.53.174`, valid Let's Encrypt cert so the JSSE-based Java crawlers
  validate the chain). Same `website-content/` the agents were run against.
- **Date:** 2026-05-31, all three crawls within 23:13–23:19 UTC.
- **Server:** `agent-scraper` (fingerproxy-based), logging to
  `/home/ubuntu/app/logs/requests-2026-05-31.jsonl`.
- **Attribution:** the server is on a public domain and sees background
  scanner/bot traffic, and all three crawls egress from one source IP, so
  records are partitioned by **User-Agent** (each crawler self-identifies
  distinctly) intersected with the per-crawl **UTC time window**. The raw
  server log (which also contains unrelated third-party traffic) is *not*
  stored here — only the filtered per-crawler records.

Reproduce a crawl (see [`baselines/<crawler>/`](../../baselines/) for setup):

```sh
# Scrapy (local, via nix-shell)
cd baselines/scrapy && TARGET_URL=https://uxbehaviorsuite.com/ PAGECOUNT=20 nix-shell --run ./run.sh
# Heritrix (Docker)
cd baselines/heritrix && TARGET_URL=https://uxbehaviorsuite.com/ DWELL_SECONDS=60 ./run.sh
# Nutch (Docker)
cd baselines/nutch && TARGET_URL=https://uxbehaviorsuite.com/ ./run.sh
```

## Captured fingerprints at a glance

| Crawler  | Reqs | ja3 (distinct)        | HTTP | HTTP/2 | Notable header behaviour |
|----------|------|-----------------------|------|--------|--------------------------|
| Scrapy   | 11   | `4b55d303…`           | 1.1  | none   | header order **unstable** (5 orderings/11 reqs) |
| Heritrix | 6    | `57128db2…`,`adbab9d5…`| 1.0  | none   | **HTTP/1.0**; two distinct ClientHellos (JSSE) |
| Nutch    | 8    | `25e40fc2…`           | 1.1  | none   | sends **Accept-Charset** (OkHttp); unique order |

> **Page coverage is bounded by each run's config, not by what the crawler
> could reach** — so `paths_fetched` is not directly comparable across the
> three. Scrapy traversed the full site (all 7 content pages). Heritrix
> early-stops once it confirms ≥2 followed links, so it only reached
> `subscribe-v1.html` among the content pages. Nutch's `topN=5`×2-rounds cap
> left `s2`/`s3`/`index.html` unfetched. The TLS/HTTP-header fingerprint is
> per-connection and fully captured in every case; only the *set of pages* is
> truncated. See each `capture.json`'s `coverage_note`. (For page-level parity
> with the agents' 30-trial page set, raise the per-crawler limits and re-run.)

Signals shared across all three baselines, and distinct from real browsers /
`curl` hitting the same server (which negotiated HTTP/2):

- **No HTTP/2** — every request is HTTP/1.0 or HTTP/1.1, so the Akamai HTTP/2
  fingerprint is empty.
- **No JS execution** — none of them run `logger.js`, so they produce **zero**
  interaction events (there is no `interactions-2026-05-31.jsonl` for the
  crawl window). Request fingerprints only.

See each `capture.json` for the exact ja3/ja4 hashes, full header orderings,
and paths fetched.
