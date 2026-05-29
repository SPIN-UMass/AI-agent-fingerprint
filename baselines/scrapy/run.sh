#!/usr/bin/env bash
# Run the Scrapy baseline smoke crawl against quotes.toscrape.com.
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

# CLOSESPIDER_PAGECOUNT counts ALL responses, including the robots.txt fetch
# and the httpbin reflection. 6 leaves room for robots.txt + a few real pages
# (so /page/2/ is demonstrably fetched) + the httpbin check.
scrapy runspider quotes_spider.py \
  -O output/quotes.jl \
  -s ROBOTSTXT_OBEY=True \
  -s CLOSESPIDER_PAGECOUNT=6 \
  -s LOG_FILE=logs/crawl.log \
  -s LOG_LEVEL=DEBUG

echo "Crawl complete."
echo "  Items: output/quotes.jl"
echo "  Log:   logs/crawl.log"
echo
echo "Observed outgoing User-Agent:"
grep -m1 "UA sent (wire value)" logs/crawl.log || true
