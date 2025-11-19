import time
import json
import random
from pathlib import Path

RUN_ID = "421AECE6"
ISLAND_ID = 0
TELEMETRY_DIR = Path(".turbo_gepa/telemetry")
TELEMETRY_DIR.mkdir(parents=True, exist_ok=True)

start_time = time.time() - 600 # Started 10 mins ago

data = []
for i in range(600):
    t = start_time + i
    eps = 5 + random.random() * 10
    lat = 1.5 + random.random() * 2.0
    
    # Simulate a queue draining
    q_ready = max(0, 200 - i * 0.3) + random.randint(0, 10)
    
    snap = {
        "timestamp": t,
        "eval_rate_eps": eps,
        "mutation_rate_mps": eps * 0.5,
        "inflight_requests": 20 + random.randint(-2, 5),
        "concurrency_limit": 25,
        "semaphore_utilization": 0.8 + random.random() * 0.2,
        "queue_ready": int(q_ready),
        "queue_mutation": random.randint(0, 5),
        "queue_replay": 0,
        "straggler_count": 0,
        "latency_p50": lat,
        "latency_p95": lat * 1.5,
        "error_rate": 0.01,
        "run_id": RUN_ID,
        "island_id": ISLAND_ID
    }
    data.append(snap)

# Write to file
outfile = TELEMETRY_DIR / f"telemetry_{RUN_ID}_{ISLAND_ID}.json"
outfile.write_text(json.dumps(data))
print(f"Generated mock telemetry at {outfile}")
