#!/usr/bin/env bash
#
# Heritrix 3 baseline smoke test for the SPIN AI-agent-fingerprint study.
#
# Builds a pinned Heritrix 3.15.0 image, starts the server-driven crawler,
# drives the REST API (create -> build -> launch -> unpause), lets it crawl
# https://quotes.toscrape.com/ for a few seconds, captures real evidence
# (crawl.log + the sent User-Agent from the WARC request records) into the
# gitignored output/ dir, then tears everything down.
#
# Usage:  ./run.sh           # build image if missing, then crawl
#         ./run.sh --rebuild # force a rebuild of the image first
#
# All crawl artifacts go to baselines/heritrix/output/ (gitignored). Nothing is
# written elsewhere in the repo and no container is left running.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="spin-heritrix:3.15.0"
CONTAINER="spin-heritrix"
JOB="spin-baseline"
OUT="${HERE}/output"
ENGINE="https://localhost:8443/engine"
CURL=(curl -ksS --anyauth -u admin:admin --location)

DWELL_SECONDS="${DWELL_SECONDS:-40}"

log() { printf '\n=== %s ===\n' "$*"; }

cleanup() {
  # Always remove the container so nothing is left running.
  docker rm -f "${CONTAINER}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

mkdir -p "${OUT}"

# --- 0. (re)build the pinned image ---------------------------------------
if [[ "${1:-}" == "--rebuild" ]] || ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
  log "Building ${IMAGE}"
  docker build -t "${IMAGE}" "${HERE}"
fi

# --- 1. start the server-driven crawler ----------------------------------
# Remove any stale container, then start fresh. -b / binds all interfaces so
# host curl can reach the REST API through the port map.
docker rm -f "${CONTAINER}" >/dev/null 2>&1 || true
log "Starting Heritrix container"
docker run -d --name "${CONTAINER}" -p 8443:8443 "${IMAGE}" >/dev/null

# Copy our job config into the container at the canonical jobs/<JOB>/ path.
docker exec "${CONTAINER}" mkdir -p "/opt/heritrix/jobs/${JOB}"
docker cp "${HERE}/crawler-beans.cxml" "${CONTAINER}:/opt/heritrix/jobs/${JOB}/crawler-beans.cxml"

# --- 2. wait for the REST API to come up ---------------------------------
# The JVM + JSSE listener take a moment to start accepting TLS. A bare curl
# probe can hit it mid-handshake and get "broken pipe" (curl exit 35), which
# curl's own --retry-connrefused does NOT cover. So we poll in shell and treat
# ANY non-zero curl exit as "not ready yet".
log "Waiting for REST API"
ready=0
for _ in $(seq 1 60); do
  if "${CURL[@]}" --max-time 4 -o /dev/null "${ENGINE}" >/dev/null 2>&1; then
    ready=1; break
  fi
  sleep 1
done
[[ "${ready}" == "1" ]] || { echo "REST API never came up"; docker logs "${CONTAINER}" | tail -30; exit 1; }
echo "REST API is up."

# --- 3. register the job dir with the engine -----------------------------
log "Adding job ${JOB}"
"${CURL[@]}" -d "action=add&addpath=/opt/heritrix/jobs/${JOB}" "${ENGINE}" >/dev/null

# --- 4. build the Spring context from crawler-beans.cxml -----------------
log "Building job"
"${CURL[@]}" -d "action=build" "${ENGINE}/job/${JOB}" >/dev/null

# --- 5. launch (starts paused: pauseAtStart defaults to true) ------------
log "Launching job"
"${CURL[@]}" -d "action=launch" "${ENGINE}/job/${JOB}" >/dev/null

# --- 6. unpause so the toe threads actually fetch ------------------------
log "Unpausing job"
"${CURL[@]}" -d "action=unpause" "${ENGINE}/job/${JOB}" >/dev/null

# --- 7. dwell while it crawls a handful of pages -------------------------
# Heritrix is polite (~3s/host). We don't just want the seed + embeds; we want
# proof it FOLLOWS navigational links. In crawl.log, field 5 is the discovery
# path: an "L" there means the URI was discovered as a Link extracted from page
# HTML (vs "P"=seed, "E"=embed/css, "I"=speculative). Those navigational links
# (e.g. /login, /author/..., /page/2/) only land ~18-25s in, so we keep crawling
# until at least 2 such L-path fetches of quotes.toscrape.com appear. Poll the
# crawl.log instead of a blind foreground sleep.
log "Crawling for up to ${DWELL_SECONDS}s (waiting for followed links)"
deadline=$(( $(date +%s) + DWELL_SECONDS ))
while [[ $(date +%s) -lt ${deadline} ]]; do
  # Count quotes.toscrape.com fetches whose discovery-path (field 5) contains L.
  # awk exits 0 even on no match; tr keeps n a single clean integer for the test.
  links=$(docker exec "${CONTAINER}" sh -c \
        "awk '\$4 ~ /quotes\.toscrape\.com/ && \$5 ~ /L/' /opt/heritrix/jobs/${JOB}/latest/logs/crawl.log 2>/dev/null | wc -l" \
        | tr -dc '0-9')
  links=${links:-0}
  printf 'followed (L-path) quotes.toscrape.com fetches so far: %s\r' "${links}"
  [[ "${links}" -ge 2 ]] && { echo; echo "link-following confirmed, stopping early."; break; }
  sleep 1
done
echo

# --- 8. terminate the job FIRST so the WARC writer flushes & closes -------
# While a job runs the WARC file is a partial ".gz.open" stream that zcat can't
# fully read; terminating finalizes it so the request records are readable.
log "Terminating job (flushes WARC)"
"${CURL[@]}" -d "action=terminate" "${ENGINE}/job/${JOB}" >/dev/null 2>&1 || true
sleep 3

# --- 9. capture evidence to gitignored output/ BEFORE teardown -----------
log "Capturing evidence to ${OUT}"
JOBROOT="/opt/heritrix/jobs/${JOB}"
docker exec "${CONTAINER}" sh -c "cat ${JOBROOT}/latest/logs/crawl.log" > "${OUT}/crawl.log" 2>/dev/null || true

# The actual SENT User-Agent lives in the WARC HTTP request records (and the
# robots.txt request), NOT in crawl.log. Extract it from the gzipped WARCs.
docker exec "${CONTAINER}" sh -c \
  "find ${JOBROOT} -name '*.warc.gz' -exec zcat {} \; 2>/dev/null | tr -d '\r' | grep -ai '^User-Agent:' | sort -u" \
  > "${OUT}/observed-user-agent.txt" 2>/dev/null || true

echo "--- crawl.log (quotes.toscrape.com fetches) ---"
grep -E 'quotes\.toscrape\.com|robots\.txt' "${OUT}/crawl.log" | head -20 || true
echo "--- observed User-Agent (sent; from WARC request records) ---"
cat "${OUT}/observed-user-agent.txt" || true

# --- 10. teardown the job, then remove the container ---------------------
log "Tearing down job"
"${CURL[@]}" -d "action=teardown"  "${ENGINE}/job/${JOB}" >/dev/null 2>&1 || true

log "Done. Evidence in ${OUT}/  (container will be removed on exit)."
