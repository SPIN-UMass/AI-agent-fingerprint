import json
import os
from datetime import datetime
from pathlib import Path
from datetime import timezone


base = './agent_transcripts/'

prompt_files = [
    f for f in os.listdir(base)
    if os.path.isfile(os.path.join(base, f)) and f.endswith(".json")
]


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
    for promt in prompt_files:
        date = (promt.split("__")[2]).split("T")[0]
        # if date == "2026-04-19":
        #     continue
        
        traffics = []
        with open(f"logs/requests-{date}.jsonl", 'r', encoding='utf-8') as f:
            for line in f:
                traffics.append(json.loads(line))

        event_logs = []
        with open(f"logs/interactions-{date}.jsonl", 'r', encoding='utf-8') as f:
            for line in f:
                event_logs.append(json.loads(line))
                
        agent = promt.split('__')[0]
        trial = promt.split('__')[1]

        with open(f"{base}{promt}", 'r', encoding='utf-8') as file:
            transcripts = json.load(file)

        finished_time = datetime.fromisoformat(transcripts["timestamp_utc"])
        if agent == "autogen_websurfer":
            starting_time = datetime.fromisoformat(transcripts["transcript"][0]["timestamp_utc"])
        elif agent == "skyvern":
            s = transcripts["transcript"]["created_at"]
            dt = datetime.strptime(
                s,
                "datetime.datetime(%Y, %m, %d, %H, %M, %S, %f)"
            )
            starting_time = dt.replace(tzinfo=timezone.utc)
        elif agent == "browser_use":
            step_start_time = transcripts["transcript"]["history"][0]["metadata"]["step_start_time"]
            starting_time = datetime.fromtimestamp(step_start_time, tz=timezone.utc)
        elif agent == "claude_computer_use":
            starting_time = datetime.fromisoformat(transcripts["transcript"][0]["started_at_utc"])
        elif agent == "gemini_computer_use":
            starting_time = datetime.fromisoformat(transcripts["transcript"][0]["started_at_utc"])
            

        traffic_results = []
        for record in traffics:
            ts = parse_ts(record["timestamp"])
            
            if starting_time <= ts <= finished_time:
                traffic_results.append(record)

        event_log_results = []
        for record in event_logs:
            for event in record.get("batch", []):
                ts = parse_ts(event["t"])

                if starting_time <= ts <= finished_time:
                    event_log_results.append(record)
                    break  # avoid duplicates

        
        Path(f"splitted_traces/{agent}/{trial}").mkdir(parents=True, exist_ok=True)
        with open(f"splitted_traces/{agent}/{trial}/requests.jsonl", 'w', encoding='utf-8') as f:
            for entry in traffic_results:
                f.write(json.dumps(entry) + '\n')

        with open(f"splitted_traces/{agent}/{trial}/interactions.jsonl", 'w', encoding='utf-8') as f:
            for entry in event_log_results:
                f.write(json.dumps(entry) + '\n')