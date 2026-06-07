#!/usr/bin/env bash
#
# Run N sequential, NON-OVERLAPPING trials of one baseline crawler against the
# fingerprint capture server, recording each trial's UTC [start,end] window so
# the server-side records can be attributed per trial afterwards
# (see attribute_trials.py).
#
# Crawlers are run one at a time, sequentially -- never concurrently -- with an
# idle gap between trials, so no two trial windows can ever intersect. This is
# the property the per-trial attribution relies on.
#
# Usage:
#   baselines/run_trials.sh <scrapy|heritrix|nutch> <trial-spec> [gap_seconds]
#
#     trial-spec   which trial numbers to run: "1", "2-30", or "5 12 27"
#     gap_seconds  idle gap between trials (default 15; must exceed clock skew)
#
#   TARGET_URL  crawl target (default https://uxbehaviorsuite.com/)
#   PAGECOUNT   Scrapy CLOSESPIDER_PAGECOUNT (default 20)
#
# Writes (gitignored, under baselines/_trials/):
#   <crawler>.manifest.tsv     trial<TAB>start<TAB>end<TAB>exit_code
#                              (appended; the LAST line for a trial wins, so a
#                               re-run of a failed trial simply supersedes it)
#   <crawler>/trial-NNN.log    per-trial crawler console output
#
# NOT set -e: one failed trial must not abort the whole batch. Exit codes are
# recorded; "success" is judged later by attributed record count, not rc.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_URL="${TARGET_URL:-https://uxbehaviorsuite.com/}"
export TARGET_URL

CRAWLER="${1:?usage: run_trials.sh <scrapy|heritrix|nutch> <trial-spec> [gap]}"
SPEC="${2:?missing trial-spec (e.g. \"2-30\")}"
GAP="${3:-15}"

WORK="${HERE}/_trials"
mkdir -p "${WORK}/${CRAWLER}"
MANIFEST="${WORK}/${CRAWLER}.manifest.tsv"

# Expand a trial-spec ("1", "2-30", "5 12 27") into a list of integers.
expand_spec() {
  local tok lo hi
  for tok in $1; do
    if [[ "$tok" == *-* ]]; then
      lo="${tok%-*}"; hi="${tok#*-}"
      seq "$lo" "$hi"
    else
      echo "$tok"
    fi
  done
}
TRIALS="$(expand_spec "$SPEC")"

ts_now() { date -u +%Y-%m-%dT%H:%M:%S.%NZ; }

# One crawl invocation per crawler, each wrapped in `timeout` so a hung
# docker run / nix realize cannot stall an unattended batch. The crawlers are
# already self-bounding (Heritrix dwell, Nutch round count, Scrapy pagecount);
# the timeout is only a backstop.
run_one() {
  case "$CRAWLER" in
    scrapy)
      ( cd "${HERE}/scrapy" \
          && PAGECOUNT="${PAGECOUNT:-20}" timeout 150 nix-shell --run "./run.sh" ) ;;
    heritrix)
      ( cd "${HERE}/heritrix" && timeout 240 ./run.sh ) ;;
    nutch)
      ( cd "${HERE}/nutch" && timeout 300 ./run.sh ) ;;
    *)
      echo "unknown crawler: ${CRAWLER} (expected scrapy|heritrix|nutch)" >&2
      exit 2 ;;
  esac
}

total="$(echo "$TRIALS" | wc -w)"
i=0
echo ">>> [${CRAWLER}] running ${total} trial(s) against ${TARGET_URL} (gap=${GAP}s)"
for n in $TRIALS; do
  i=$((i + 1))
  nnn="$(printf '%03d' "$n")"
  log="${WORK}/${CRAWLER}/trial-${nnn}.log"
  start="$(ts_now)"
  echo ">>> [${CRAWLER} ${i}/${total}] trial ${nnn} start ${start}"
  run_one >"$log" 2>&1
  rc=$?
  end="$(ts_now)"
  printf '%s\t%s\t%s\t%s\n' "$nnn" "$start" "$end" "$rc" >> "$MANIFEST"
  echo "    trial ${nnn} done rc=${rc}  window=[${start} .. ${end}]"
  if [[ "$i" -lt "$total" ]]; then sleep "$GAP"; fi
done
echo ">>> [${CRAWLER}] batch complete: ${total} trial(s); manifest=${MANIFEST}"
