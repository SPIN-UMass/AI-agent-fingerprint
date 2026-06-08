import json
import os
from datetime import datetime
from pathlib import Path
from datetime import timezone, timedelta
from zoneinfo import ZoneInfo


base = './agent_transcripts/'

def parse_ts(ts):
    # Convert Z → +00:00 and trim nanoseconds to microseconds
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    # keep only first 6 digits of fractional seconds
    if "." in ts:
        date_part, frac = ts.split(".")
        frac, tz = frac[:6], frac[6:]
        ts = f"{date_part}.{frac}{tz}"
    return datetime.fromisoformat(ts)

if __name__=="__main__":
    year = "2026"
    agent = "human"
    eastern = ZoneInfo("America/New_York")

    with open('human_timestamp.txt', 'r') as file:
        for count, promt in enumerate(file, start=1):
            trial = f"trial-{count:03d}"
            date_part, start_time, _, end_time = promt.split()
            
            # Build full datetime strings
            start_dt = datetime.strptime(f"{year}/{date_part} {start_time}", "%Y/%m/%d %H:%M")
            end_dt = datetime.strptime(f"{year}/{date_part} {end_time}", "%Y/%m/%d %H:%M")

            start_dt = start_dt.replace(tzinfo=eastern)
            end_dt = end_dt.replace(tzinfo=eastern)

            starting_time = start_dt.astimezone(ZoneInfo("UTC"))
            finished_time = (end_dt + timedelta(seconds=59)).astimezone(ZoneInfo("UTC"))

            date = starting_time.strftime("%Y-%m-%d")
            
            traffics = []
            with open(f"logs/requests-{date}.jsonl", 'r', encoding='utf-8') as f:
                for line in f:
                    traffics.append(json.loads(line))

            event_logs = []
            with open(f"logs/interactions-{date}.jsonl", 'r', encoding='utf-8') as f:
                for line in f:
                    event_logs.append(json.loads(line))

            traffic_results = []
            for record in traffics:
                ts = parse_ts(record["timestamp"])
                
                if starting_time <= ts < finished_time:
                    traffic_results.append(record)

            event_log_results = []
            for record in event_logs:
                for event in record.get("batch", []):
                    ts = parse_ts(event["t"])

                    if starting_time <= ts < finished_time:
                        event_log_results.append(record)
                        break  # avoid duplicates

            
            Path(f"splitted_traces/{agent}/{trial}").mkdir(parents=True, exist_ok=True)
            with open(f"splitted_traces/{agent}/{trial}/requests.jsonl", 'w', encoding='utf-8') as f:
                for entry in traffic_results:
                    f.write(json.dumps(entry) + '\n')

            with open(f"splitted_traces/{agent}/{trial}/interactions.jsonl", 'w', encoding='utf-8') as f:
                for entry in event_log_results:
                    f.write(json.dumps(entry) + '\n')