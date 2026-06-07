#!/usr/bin/env python3
"""Attribute server-side fingerprint records to per-trial crawl windows.

Reads a run_trials.sh manifest (trial / start / end / exit_code; the LAST line
for a trial wins, so re-runs supersede earlier attempts) plus one or more
server request logs (requests-YYYY-MM-DD.jsonl). For each trial it selects the
records whose timestamp falls in [start - PAD, end + PAD] AND whose HTTP
user_agent contains the crawler's UA marker, and writes them verbatim to:

    <dest>/<crawler>/trial-NNN/requests.jsonl
    <dest>/<crawler>/trial-NNN/interactions.jsonl   (always empty: no JS engine)

The crawls are sequential and non-overlapping (run_trials.sh enforces the gap),
so the window cleanly isolates a trial; the UA marker removes co-resident
background-scanner traffic. Source IP is used as a cross-check, not a filter:
every record is expected to carry one of the observed egress IPs; any that does
not is surfaced in the summary (NAT rotation, or -- implausibly, given the
study-specific UAs -- a scanner spoofing the marker), never silently dropped.

Emits <dest>/<crawler>/trials_summary.json and prints a human-readable summary
(per-trial counts, empty trials to re-run, IP anomalies) to stdout.

Usage:
  attribute_trials.py --crawler scrapy --ua 'Scrapy/2.14.1' \
      --egress 73.119.206.174 [--egress ...] \
      --manifest baselines/_trials/scrapy.manifest.tsv \
      --dest fingerprint_data/baselines \
      --log /tmp/serverlogs/requests-2026-06-07.jsonl [--log ...] \
      [--pad 5]
"""
import argparse
import json
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path


def parse_ts(ts: str) -> datetime:
    """Parse an RFC3339 timestamp, trimming sub-microsecond precision.

    Matches the convention in fingerprint_data/split_traces.py: 'Z' -> +00:00
    and fractional seconds truncated to 6 digits (datetime's limit).
    """
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    if "." in ts:
        date_part, frac = ts.split(".")
        # frac may carry a trailing timezone offset after the digits
        i = 0
        while i < len(frac) and frac[i].isdigit():
            i += 1
        digits, tail = frac[:i], frac[i:]
        ts = f"{date_part}.{digits[:6]}{tail}"
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def read_manifest(path: Path):
    """Return {trial_int: (start_dt, end_dt, exit_code)} keeping the last line."""
    trials = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        n = int(parts[0])
        start = parse_ts(parts[1])
        end = parse_ts(parts[2])
        rc = parts[3] if len(parts) > 3 else ""
        trials[n] = (start, end, rc)  # later lines overwrite earlier ones
    return trials


def load_records(log_paths):
    """Parse each server log once into (dt, ua, source_ip, original_line, rec)."""
    out = []
    for p in log_paths:
        with open(p, "r", encoding="utf-8") as f:
            for raw in f:
                raw = raw.rstrip("\n")
                if not raw.strip():
                    continue
                try:
                    rec = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                ts = rec.get("timestamp")
                if not ts:
                    continue
                try:
                    dt = parse_ts(ts)
                except ValueError:
                    continue
                ua = (rec.get("http") or {}).get("user_agent", "") or ""
                ip = rec.get("source_ip", "") or ""
                out.append((dt, ua, ip, raw, rec))
    out.sort(key=lambda r: r[0])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crawler", required=True)
    ap.add_argument("--ua", required=True, help="UA substring marker for this crawler")
    ap.add_argument("--egress", action="append", default=[],
                    help="observed egress IP(s); repeatable. Cross-check only.")
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--dest", required=True, type=Path,
                    help="base dir; trials land in <dest>/<crawler>/trial-NNN/")
    ap.add_argument("--log", action="append", required=True, dest="logs",
                    help="server request log(s); repeatable")
    ap.add_argument("--pad", type=float, default=5.0,
                    help="seconds of slack added to each side of a window")
    args = ap.parse_args()

    egress = set(args.egress)
    pad = timedelta(seconds=args.pad)
    trials = read_manifest(args.manifest)
    records = load_records(args.logs)

    crawler_dir = args.dest / args.crawler
    summary_trials = []
    empties = []
    anomalies = []

    for n in sorted(trials):
        start, end, rc = trials[n]
        lo, hi = start - pad, end + pad
        matched = [r for r in records if lo <= r[0] <= hi and args.ua in r[1]]

        tdir = crawler_dir / f"trial-{n:03d}"
        tdir.mkdir(parents=True, exist_ok=True)
        with open(tdir / "requests.jsonl", "w", encoding="utf-8") as f:
            for _, _, _, raw, _ in matched:
                f.write(raw + "\n")
        # No JS engine -> no logger.js beacons. Empty for layout parity with
        # the AI-agent traces; window-attributing interactions here would
        # falsely capture unrelated background-browser beacons.
        (tdir / "interactions.jsonl").write_text("")

        ips = Counter(r[2] for r in matched)
        non_egress = sorted(ip for ip in ips if egress and ip not in egress)
        ja3 = sorted({(r[4].get("tls") or {}).get("ja3_hash") for r in matched} - {None})
        ja4 = sorted({(r[4].get("tls") or {}).get("ja4_hash") for r in matched} - {None})
        protos = sorted({(r[4].get("http") or {}).get("protocol") for r in matched} - {None})
        akamai = sorted({(r[4].get("http2") or {}).get("akamai_fingerprint", "") for r in matched})
        paths = [(r[4].get("http") or {}).get("path") for r in matched]

        t = {
            "trial": n,
            "window_utc": {"start": start.isoformat().replace("+00:00", "Z"),
                           "end": end.isoformat().replace("+00:00", "Z")},
            "exit_code": rc,
            "record_count": len(matched),
            "source_ips": dict(ips),
            "non_egress_ips": non_egress,
            "ja3_hashes": ja3,
            "ja4_hashes": ja4,
            "protocols": protos,
            "http2_akamai_fingerprints": akamai,
            "paths_fetched": paths,
        }
        summary_trials.append(t)
        if len(matched) == 0:
            empties.append(n)
        if non_egress:
            anomalies.append((n, non_egress))

    summary = {
        "crawler": args.crawler,
        "ua_marker": args.ua,
        "observed_egress_ips": sorted(egress),
        "pad_seconds": args.pad,
        "source_logs": [str(p) for p in args.logs],
        "trials_run": sorted(trials),
        "trial_count": len(trials),
        "empty_trials": empties,
        "ja3_union": sorted({h for t in summary_trials for h in t["ja3_hashes"]}),
        "ja4_union": sorted({h for t in summary_trials for h in t["ja4_hashes"]}),
        "protocols_union": sorted({p for t in summary_trials for p in t["protocols"]}),
        "ip_anomaly_trials": [n for n, _ in anomalies],
        "trials": summary_trials,
    }
    crawler_dir.mkdir(parents=True, exist_ok=True)
    (crawler_dir / "trials_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    # ---- human-readable report ----
    print(f"[{args.crawler}] {len(trials)} trial(s) attributed -> {crawler_dir}")
    for t in summary_trials:
        flag = "  <-- IP ANOMALY" if t["non_egress_ips"] else ""
        print(f"  trial-{t['trial']:03d}: {t['record_count']:3d} rec  "
              f"ja3={','.join(t['ja3_hashes']) or '-'}  "
              f"ips={','.join(t['source_ips']) or '-'}{flag}")
    print(f"  ja3_union: {summary['ja3_union']}")
    print(f"  protocols: {summary['protocols_union']}")
    if empties:
        print(f"  EMPTY TRIALS (re-run): {empties}")
    else:
        print("  all trials non-empty.")
    if anomalies:
        for n, ips in anomalies:
            print(f"  IP ANOMALY trial-{n:03d}: non-egress {ips}")


if __name__ == "__main__":
    main()
