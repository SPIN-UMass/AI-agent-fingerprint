#!/usr/bin/env bash
#
# SPIN AI-agent-fingerprint -- Apache Nutch 1.22 baseline crawler smoke test.
#
# Runs the Nutch batch pipeline (inject -> [generate -> fetch -> parse -> updatedb] x2)
# against https://quotes.toscrape.com/ inside the pinned official Docker image, using
# our committed configs (nutch-site.xml, regex-urlfilter.txt, log4j2.xml, urls/seed.txt).
#
# All crawl state + logs go to gitignored paths (./output, ./logs). The container is
# removed on exit. We capture the ACTUAL default User-Agent and the /robots.txt request
# from the DEBUG fetch log.
#
# Usage:  ./run.sh
set -euo pipefail

IMAGE="apache/nutch:release-1.22"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NUTCH_HOME="/root/nutch_source/runtime/local"

OUT="${HERE}/output"
LOGS="${HERE}/logs"
rm -rf "${OUT}" "${LOGS}"
mkdir -p "${OUT}" "${LOGS}"

HOST_UID="$(id -u)"
HOST_GID="$(id -g)"

echo "==> Using image: ${IMAGE}"
docker image inspect "${IMAGE}" >/dev/null 2>&1 || docker pull "${IMAGE}"

# Mount our individual config files over the image's conf dir (read-only), the seed
# dir, and writable output/logs dirs. We do NOT overlay the whole conf dir so the
# image's nutch-default.xml and the other plugin config files stay intact.
# Run as root inside the container (the image's NUTCH_HOME is root-owned and Nutch
# writes scratch files there); we chown the bind-mounted output/logs back to the host
# user at the end so nothing root-owned is left in the repo tree.
docker run --rm \
  --name spin-nutch-smoke \
  -e HOST_UID="${HOST_UID}" \
  -e HOST_GID="${HOST_GID}" \
  -v "${HERE}/nutch-site.xml:${NUTCH_HOME}/conf/nutch-site.xml:ro" \
  -v "${HERE}/regex-urlfilter.txt:${NUTCH_HOME}/conf/regex-urlfilter.txt:ro" \
  -v "${HERE}/log4j2.xml:${NUTCH_HOME}/conf/log4j2.xml:ro" \
  -v "${HERE}/urls:${NUTCH_HOME}/urls:ro" \
  -v "${OUT}:${NUTCH_HOME}/output" \
  -v "${LOGS}:${NUTCH_HOME}/logs" \
  --entrypoint /bin/bash \
  "${IMAGE}" -c '
    set -e
    cd "'"${NUTCH_HOME}"'"
    CRAWLDB=output/crawldb
    SEGMENTS=output/segments

    echo "===== inject seeds ====="
    bin/nutch inject "$CRAWLDB" urls

    for round in 1 2; do
      echo "===== round ${round}: generate ====="
      # -topN keeps each batch tiny; returns nonzero when nothing to fetch.
      if ! bin/nutch generate "$CRAWLDB" "$SEGMENTS" -topN 5 > /tmp/gen.out 2>&1; then
        cat /tmp/gen.out
        echo "Nothing left to generate; stopping early."
        break
      fi
      cat /tmp/gen.out
      SEG="$SEGMENTS/$(ls -t "$SEGMENTS" | head -1)"
      echo "===== round ${round}: fetch ${SEG} ====="
      bin/nutch fetch "$SEG" -threads 1
      echo "===== round ${round}: parse ${SEG} ====="
      bin/nutch parse "$SEG"
      echo "===== round ${round}: updatedb ====="
      bin/nutch updatedb "$CRAWLDB" "$SEG"
    done

    echo "===== readdb stats ====="
    bin/nutch readdb "$CRAWLDB" -stats || true

    echo "===== server-side UA check: one fetch of httpbin.org/headers ====="
    # Independent cross-check of the UA the server actually receives. httpbin.org
    # echoes the request headers in its JSON body, which Nutch stores in the segment;
    # readseg -dump lets us read the User-Agent the server saw. This is best-effort
    # (the authoritative UA is the "http.agent = ..." line already in hadoop.log), so
    # the whole block is non-fatal and uses -noFilter to bypass our host-pinned filter.
    set +e
    (
      mkdir -p output/uaurls
      echo "https://httpbin.org/headers" > output/uaurls/seed.txt
      bin/nutch inject output/uacheckdb output/uaurls -noFilter
      bin/nutch generate output/uacheckdb output/uasegments -topN 1 -noFilter
      UASEG="output/uasegments/$(ls -t output/uasegments | head -1)"
      bin/nutch fetch "$UASEG" -threads 1
      bin/nutch parse "$UASEG"
      echo "----- dumping httpbin response (shows the User-Agent header the server saw) -----"
      bin/nutch readseg -dump "$UASEG" output/uadump -nofetch -nogenerate -noparse -noparsedata
      grep -aiE "user-agent" output/uadump/dump | head -5
    ) || echo "(httpbin cross-check skipped/failed; authoritative UA is in logs/hadoop.log)"
    set -e

    # Always hand artifacts back to the host user so the repo tree has no root-owned files.
    chown -R "${HOST_UID}:${HOST_GID}" output logs 2>/dev/null || true
  ' 2>&1 | tee "${LOGS}/run-console.log"

echo
echo "==================================================================="
echo "Crawl finished. Logs: ${LOGS}/hadoop.log and ${LOGS}/run-console.log"
echo "==================================================================="
echo
echo "----- composed http.agent (authoritative UA actually used) -----"
grep -h "http.agent =" "${LOGS}/hadoop.log" | head -3 || true
echo
echo "----- quotes.toscrape.com fetches (incl. /robots.txt at DEBUG) -----"
grep -hE "quotes\.toscrape\.com" "${LOGS}/hadoop.log" | grep -iE "fetch|robots| - HTTP" | head -25 || true
