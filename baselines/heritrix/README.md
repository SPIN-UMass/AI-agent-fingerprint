# Heritrix 3 baseline crawler

[Heritrix](https://github.com/internetarchive/heritrix3) is the Internet
Archive's open-source, web-scale, archival-quality web crawler. This directory
sets it up reproducibly as a baseline for the SPIN AI-agent-fingerprint study
and records its **actual** default self-identification.

- **Crawler:** Heritrix 3
- **Version (pinned):** `3.15.0` (latest release as of 2026-05-26;
  Maven coordinates `org.archive.heritrix:heritrix`). Requires **Java 17+**
  (Heritrix >= 3.15.0 uses Spring Framework 6.1, which requires Java 17).
- **Install method:** pinned Docker image built from `eclipse-temurin:17-jdk`.
  There is **no official Heritrix Docker image** (it is an Internet Archive
  project, not Apache), so the `Dockerfile` downloads the official distribution
  zip from Maven Central *inside the image build* (with a SHA-1 checksum check)
  and unzips it there. Nothing is vendored into the repo.

  ```
  https://repo1.maven.org/maven2/org/archive/heritrix/heritrix/3.15.0/heritrix-3.15.0-dist.zip
  sha1: bc25b1ec3fe030ae74d42909b9ea599d08204564
  ```

## Files

| File | Purpose |
|------|---------|
| `Dockerfile` | Pinned Temurin-JDK-17 + Heritrix 3.15.0 dist image. |
| `crawler-beans.cxml` | Job config = the bundled `profile-crawler-beans.cxml` **verbatim**, with only three changes (see below). |
| `run.sh` | Build, drive the REST API, crawl, capture evidence, tear down. |

`crawler-beans.cxml` is the stock Heritrix "basic" profile (a Spring context).
The only edits to the default profile are:

1. `metadata.operatorContactUrl` -> `https://example.org/SPIN-baseline`
   (placeholder contact URL; it appears in the User-Agent).
2. `seeds.textSource.value` -> `https://quotes.toscrape.com/` (the single seed).
3. `crawlController.maxToeThreads` -> `2` (keep the smoke test small/polite).

Everything else (scope rules, WARC writer chain, robots policy, politeness) is
left at Heritrix defaults so the observed behavior is representative.

## Build

```sh
docker build -t spin-heritrix:3.15.0 baselines/heritrix/
```

## Run (single command)

```sh
baselines/heritrix/run.sh            # builds image if missing, then crawls
baselines/heritrix/run.sh --rebuild  # force image rebuild first
```

`run.sh` is self-contained and idempotent: it (re)creates the container, drives
the full job lifecycle over the REST API, captures evidence into the gitignored
`output/` dir, and removes the container on exit (even on failure, via a trap).
No container, server, or crawl artifact is left behind in the repo tree.

## How it works: server-driven crawl over the REST API

Heritrix is **server-driven**: the container starts a Web UI / REST API on
`:8443` (HTTPS, self-signed cert, DIGEST auth). It is started with
`bin/heritrix -a admin:admin -b /` — `-b /` binds the admin server to **all**
interfaces (the default is localhost-only, which would be unreachable through
the Docker port map). `run.sh` then drives it from the host with curl:

```sh
ENGINE=https://localhost:8443/engine
CURL='curl -ksS --anyauth -u admin:admin --location'

# wait until the TLS listener accepts (poll; tolerate broken-pipe during boot)
$CURL $ENGINE

# 1. register the mounted job directory
$CURL -d "action=add&addpath=/opt/heritrix/jobs/spin-baseline" $ENGINE

# 2. build the Spring context from crawler-beans.cxml
$CURL -d "action=build"     $ENGINE/job/spin-baseline

# 3. launch (the job starts PAUSED: pauseAtStart defaults to true)
$CURL -d "action=launch"    $ENGINE/job/spin-baseline

# 4. unpause so the toe threads fetch
$CURL -d "action=unpause"   $ENGINE/job/spin-baseline

# ... crawl ~25s (polite ~3s/host) ...

# 5. terminate FIRST so the WARC writer flushes & closes the .gz
$CURL -d "action=terminate" $ENGINE/job/spin-baseline

# 6. teardown the job
$CURL -d "action=teardown"  $ENGINE/job/spin-baseline
```

All artifacts (job dir, `crawl.log`, WARCs) live inside the ephemeral container
under `/opt/heritrix/jobs/spin-baseline/`; `run.sh` `docker exec`/`docker cp`s
the evidence we need into the gitignored `output/` before teardown.

## Observed default self-identification

The crawler self-identifies via its User-Agent. Heritrix's default UA template
is `Mozilla/5.0 (compatible; heritrix/@VERSION@ +@OPERATOR_CONTACT_URL@)`, with
`@VERSION@` and `@OPERATOR_CONTACT_URL@` substituted at runtime.

The crawl-log fields (timestamp, status, size, URI, discovery-path, ...) do
**not** include request headers, so the *sent* User-Agent is read from the WARC
**HTTP request records** that Heritrix writes by default
(`HttpRequestRecordBuilder` is in the WARC writer chain):

```sh
find <jobdir> -name '*.warc.gz' -exec zcat {} \; | grep -ai '^User-Agent:'
```

**Observed sent User-Agent (from WARC request records, this crawl):**

```
Mozilla/5.0 (compatible; heritrix/3.15.0 +https://example.org/SPIN-baseline)
```

This is the actual byte string sent on the wire (verified server-side via the
WARC request record), not merely the documented template.

## robots.txt behavior

Heritrix fetches `/robots.txt` by default (robots policy `obey`). For this
target the crawl.log shows the robots fetch returning **404**, which Heritrix
treats as allow-all, so the seed and discovered links are then crawled:

```
... 404 ... https://quotes.toscrape.com/robots.txt           P  https://quotes.toscrape.com/ text/html ...
... 200 ... https://quotes.toscrape.com/                      -  -                             text/html ...
... 200 ... https://quotes.toscrape.com/static/bootstrap...css E https://quotes.toscrape.com/ text/css  ...
... 200 ... https://quotes.toscrape.com/login                 L  https://quotes.toscrape.com/ text/html ...
... 308 ... https://quotes.toscrape.com/author/Albert-Einstein L  https://quotes.toscrape.com/ text/html ...
```

Field 5 is the **discovery path**: `P` = seed, `E` = embed (CSS/img), `I` =
speculative, and `L` = a navigational **link extracted from the page HTML**.
The two `L` lines above (`/login`, `/author/Albert-Einstein`) are the proof
that Heritrix discovered and followed links out of the seed page — `run.sh`
intentionally crawls until at least two such `L` fetches appear before stopping.

(The robots.txt request is itself a WARC request record and also carries the
above User-Agent — independent confirmation.)

## TLS stack note (for the fingerprint study)

Heritrix runs on the JVM and performs HTTPS through **Apache HttpComponents
HttpClient over the JDK's built-in JSSE TLS provider** (SunJSSE). So the
TLS/JA3-style fingerprint of a Heritrix crawler is that of the Temurin/OpenJDK
17 JSSE stack, distinct from Go (`net/http`) or libcurl/OpenSSL crawlers.
