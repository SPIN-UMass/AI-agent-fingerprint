# Apache Nutch baseline crawler (the crawler family behind CCBot)

Reproducible smoke-test setup for **Apache Nutch 1.22** as a baseline crawler for the
SPIN AI-agent-fingerprint study. Runs the standard Nutch batch pipeline
(`inject -> generate -> fetch -> parse -> updatedb`) against the small Zyte sandbox
`https://quotes.toscrape.com/`, captures the crawler's actual default User-Agent, and
confirms its robots.txt behavior.

## Install method

No JVM / Maven on the host. We use the official, **pinned** Docker image:

```
apache/nutch:release-1.22
```

(Nutch 1.22, Java. `NUTCH_HOME=/root/nutch_source/runtime/local` inside the image.)
At time of writing, the `release-1.22` tag resolves to
`sha256:d0a8339f73258ba90d9da9141aa713873ee70da9d5bbcb46b294bad509f342d1` (the immutable
pin if you want one). Note the image's internal `http.agent.version` reports
`Nutch-1.22-SNAPSHOT`, which is why that token appears in the User-Agent below.
Nothing is unpacked into the repo; the image is ephemeral and the container is removed
on exit (`docker run --rm`).

## Run

```sh
./run.sh
```

That is the single reproducible command. It:

1. mounts our committed config files **over** the image's `conf/` (read-only) so the
   image's `nutch-default.xml` and other plugin configs stay intact;
2. mounts `urls/` (seed) read-only and `output/` + `logs/` read-write to **gitignored**
   host paths;
3. runs `inject`, then **2 rounds** of `generate -topN 5 / fetch / parse / updatedb`
   (round 1 fetches the seed; round 2 follows discovered on-site links), then `readdb -stats`;
4. does a best-effort server-side UA cross-check by fetching `https://httpbin.org/headers`
   (which echoes the request headers it received);
5. `chown`s artifacts back to the host user and the container is auto-removed.

All crawl state (`crawldb`, `segments`) and logs go to `baselines/nutch/output/` and
`baselines/nutch/logs/` (both gitignored). Nothing is committed except the configs below.

## Committed files (and their purpose)

| File | Purpose |
|------|---------|
| `nutch-site.xml` | Site overrides. Sets **`http.agent.name`** (Nutch refuses to fetch without it), `http.agent.url` (contact in the UA), and swaps `protocol-http` -> **`protocol-okhttp`** in `plugin.includes`. Also sets a 1s crawl delay, 1 thread/queue, and `generate.max.count=5` to keep it small. |
| `regex-urlfilter.txt` | Replaces the image default (whose last rule is `+.` = accept everything). Pins the crawl to `quotes.toscrape.com` and rejects all else, so round 2 stays on-site/small instead of wandering to external links. |
| `log4j2.xml` | Identical to the image default except the `org.apache.nutch.protocol` logger is raised to **DEBUG**, so the per-request `<url> - <proto> <code>` line (incl. `/robots.txt`) appears in `hadoop.log` as evidence. |
| `urls/seed.txt` | Single seed URL: `https://quotes.toscrape.com/`. |
| `run.sh` | The orchestration described above. |

## Observed default User-Agent (ACTUAL, from logs)

Authoritative line logged by Nutch (`o.a.n.p.o.OkHttp` — `http.agent = ...`), and
independently confirmed server-side by httpbin echoing the header it received:

```
SPIN-baseline-nutch/Nutch-1.22-SNAPSHOT (https://github.com/SPIN-UMass/AI-agent-fingerprint)
```

- Format is built by `lib-http` `HttpBase.getAgentString()`:
  `<http.agent.name>/<http.agent.version> (<http.agent.url>; ...)`.
- `http.agent.name` = `SPIN-baseline-nutch` (we set it; the only token that's *required*).
- `http.agent.version` = `Nutch-1.22-SNAPSHOT` — the **image default**, deliberately NOT
  overridden, so this is the real string the image sends (note the literal `-SNAPSHOT`).
- The robots parser matches the lowercased agent token `spin-baseline-nutch`.

## Robots.txt behavior

Nutch is RFC 9309 compliant and **fetches `/robots.txt` first** for each host before
fetching content. Confirmed in `logs/hadoop.log`:

```
DEBUG o.a.n.p.o.OkHttpResponse  https://quotes.toscrape.com/robots.txt - http/1.1 404 NOT FOUND
DEBUG o.a.n.p.h.a.HttpRobotRulesParser  Fetched robots.txt for https://quotes.toscrape.com/ with status code 404
INFO  o.a.n.p.RobotRulesParser  Checking robots.txt for the following agent names: [spin-baseline-nutch]
```

The target returns 404 for `/robots.txt`, which Nutch treats as allow-all.

## Sample fetched URLs (HTTP 200, across the 2 rounds)

```
https://quotes.toscrape.com/                       (round 1, seed)
https://quotes.toscrape.com/tag/world/page/1/      (round 2, followed link)
https://quotes.toscrape.com/tag/humor/page/1/      (round 2, followed link)
https://quotes.toscrape.com/tag/life/page/1/       (round 2, followed link)
https://quotes.toscrape.com/tag/inspirational/page/1/  (round 2, followed link)
```

`readdb -stats` reported `db_fetched: 5`, `db_unfetched: 95` (outlinks discovered).

## Note for the fingerprint study: TLS stack

This baseline uses the **`protocol-okhttp`** plugin: HTTP/TLS via **OkHttp3** running on
the **JVM's default JSSE TLS provider**. The plugin creates its TLS context with
`SSLContext.getInstance("TLS")` (see
`protocol-okhttp/.../OkHttp.java`), so the TLS ClientHello / JA3 fingerprint is that of
the JDK shipped in the `apache/nutch:release-1.22` image (OpenJDK), shaped by OkHttp3's
connection specs — distinct from a Python/`requests` or Go crawler.

### Discrepancy worth noting

The task brief described `protocol-okhttp` as the Nutch default. In this image it is
**not**: the shipped `nutch-default.xml` `plugin.includes` lists `protocol-http` (a
plain `java.net.HttpURLConnection`-based client). We explicitly switched to
`protocol-okhttp` in `nutch-site.xml` to match the study's intended OkHttp3/JSSE stack.
Both plugins share the same `lib-http` UA code, so the User-Agent above is identical
either way; only the underlying HTTP client and TLS ClientHello differ.
