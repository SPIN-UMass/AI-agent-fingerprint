#!/usr/bin/env bash
# Run the Scrapy baseline crawl.
#
# Defaults to the public quotes.toscrape.com sandbox. Override the target to
# point it at the study's fingerprint server:
#   TARGET_URL=https://uxbehaviorsuite.com/ PAGECOUNT=20 ./run.sh
#
# Enter the nix shell first, or let this script be invoked through it:
#   cd baselines/scrapy
#   nix-shell --run "./run.sh"
#
# Output (gitignored):
#   output/quotes.jl  -- scraped items, one JSON object per line
#   logs/crawl.log    -- full Scrapy DEBUG log (UA, robots.txt, fetched URLs)
set -euo pipefail

cd "$(dirname "$0")"

mkdir -p output logs

# Target + crawl size are configurable. The spider reads TARGET_URL from the
# environment at import time, so it must be exported for the scrapy process.
export TARGET_URL="${TARGET_URL:-https://quotes.toscrape.com/}"
PAGECOUNT="${PAGECOUNT:-6}"

echo "Target: ${TARGET_URL}  (CLOSESPIDER_PAGECOUNT=${PAGECOUNT})"

# CLOSESPIDER_PAGECOUNT counts ALL responses, including the robots.txt fetch
# and (for the quotes sandbox) the httpbin reflection. The default of 6 leaves
# room for robots.txt + a few real pages + the httpbin check; bump PAGECOUNT to
# traverse a larger target in full.
scrapy runspider quotes_spider.py \
  -O output/quotes.jl \
  -s ROBOTSTXT_OBEY=True \
  -s CLOSESPIDER_PAGECOUNT="${PAGECOUNT}" \
  -s LOG_FILE=logs/crawl.log \
  -s LOG_LEVEL=DEBUG

echo "Crawl complete."
echo "  Items: output/quotes.jl"
echo "  Log:   logs/crawl.log"
echo
echo "Observed outgoing User-Agent:"
grep -m1 "UA sent (wire value)" logs/crawl.log || true
