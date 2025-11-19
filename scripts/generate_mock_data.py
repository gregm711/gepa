"""
Generate rich mock data for TurboGEPA visualization.
Creates a complex evolution graph and simulates live telemetry for dashboard screenshots.

Usage:
    python scripts/generate_mock_data.py
"""

import json
import math
import os
import random
import time
import uuid
from pathlib import Path

# Config
RUN_ID = f"mock-run-{uuid.uuid4().hex[:6]}"
NUM_ISLANDS = 4
TURBO_DIR = Path(".turbo_gepa")
EVO_DIR = TURBO_DIR / "evolution"
TELEMETRY_DIR = TURBO_DIR / "telemetry"

def setup_dirs():
    EVO_DIR.mkdir(parents=True, exist_ok=True)
    TELEMETRY_DIR.mkdir(parents=True, exist_ok=True)

def generate_fingerprint():
    return uuid.uuid4().hex[:16]

def generate_graph():
    """Generate a realistic optimization graph structure."""
    nodes = []
    edges = {} # parent -> [children]
    
    # Seed
    seed_fp = generate_fingerprint()
    nodes.append({
        "fingerprint": seed_fp,
        "generation": 0,
        "quality": 0.50,
        "shard_fraction": 1.0,
        "status": "promoted",
        "prompt": "You are a helpful assistant.",
        "prompt_full": "You are a helpful assistant.",
        "current_island": 0
    })
    
    generations = 8
    current_layer = [seed_fp]
    all_fps = {seed_fp}
    
    timeline = []
    
    # Simulate evolution
    for gen in range(1, generations + 1):
        next_layer = []
        best_quality_in_gen = 0.0
        
        # Scale quality with generation: 0.5 -> 0.95
        base_quality = 0.50 + (gen / generations) * 0.45
        
        for parent_idx, parent_fp in enumerate(current_layer):
            # Determine fate of this parent
            # Spawn children - guarantee breadth
            num_children = random.randint(2, 4)
            
            children_fps = []
            for i in range(num_children):
                child_fp = generate_fingerprint()
                island = random.randint(0, NUM_ISLANDS - 1)
                
                # Simulate ASHA: chance to be pruned early
                # Add noise
                quality = min(0.99, max(0.0, random.gauss(base_quality, 0.08)))
                
                # Status & Shard logic
                status = "pruned"
                shard = 0.05
                
                # Force at least one promoted child for the first parent to ensure depth
                force_promote = (parent_idx == 0 and i == 0)
                
                if force_promote or quality > 0.85:
                    status = "promoted"
                    shard = 1.0
                    # Boost quality for the forced winner if needed
                    if force_promote:
                        quality = max(quality, base_quality + 0.05)
                elif quality > 0.70:
                    status = "other" # Stuck in middle
                    shard = 0.2
                else:
                    status = "pruned"
                    shard = 0.05
                
                best_quality_in_gen = max(best_quality_in_gen, quality)
                
                nodes.append({
                    "fingerprint": child_fp,
                    "generation": gen,
                    "quality": quality,
                    "shard_fraction": shard,
                    "status": status,
                    "prompt": f"Gen {gen} variant {i}...\n(Mock prompt content for visual)",
                    "prompt_full": f"Full prompt content for {child_fp}...",
                    "current_island": island
                })
                
                children_fps.append(child_fp)
                all_fps.add(child_fp)
                
                if status in ("promoted", "other"):
                    next_layer.append(child_fp)
            
            if children_fps:
                edges[parent_fp] = children_fps
        
        current_layer = next_layer
        
        # Add timeline entry
        timeline.append({
            "elapsed": gen * 15.0,
            "best_quality": best_quality_in_gen,
            "evaluations": len(nodes)
        })
        
        if not current_layer: # Should not happen with forced promotion
            break

    return nodes, edges, timeline

def write_evolution_file(nodes, edges, timeline):
    data = {
        "run_id": RUN_ID,
        "lineage": nodes,
        "evolution_stats": {
            "parent_children": edges,
            "mutations_generated": len(nodes),
            "unique_parents": len(edges),
            "evolution_edges": sum(len(v) for v in edges.values())
        },
        "timeline": timeline,
        "run_metadata": {
            "evaluations": len(nodes),
            "best_quality": max(n["quality"] for n in nodes),
            "best_prompt": "The winning mock prompt..."
        }
    }
    
    # Write main file
    with open(EVO_DIR / f"{RUN_ID}.json", "w") as f:
        json.dump(data, f)
        
    # Update pointer
    with open(EVO_DIR / "current.json", "w") as f:
        json.dump({"run_id": RUN_ID}, f)
        
    print(f"✅ Evolution graph written: {len(nodes)} nodes, {len(edges)} parents")

def simulate_telemetry_loop():
    print(f"🚀 Starting telemetry simulation for run {RUN_ID}...")
    print("   (Keep this running. Open the dashboard now!)")
    
    start_time = time.time()
    evals_completed = 0
    
    try:
        while True:
            t = time.time() - start_time
            
            # Simulate varying load
            base_eps = 15.0 + 10.0 * math.sin(t / 5.0)  # 5-25 EPS
            base_lat = 0.8 + 0.4 * math.sin(t / 3.0)    # 0.4-1.2s latency
            
            for island in range(NUM_ISLANDS):
                # Randomize per island
                noise = random.uniform(0.8, 1.2)
                eps = base_eps * noise / NUM_ISLANDS
                lat = base_lat * noise
                
                # Queue buildup
                q_ready = int(50 + 30 * math.sin(t / 10.0) + random.randint(-5, 5))
                
                snap = {
                    "timestamp": time.time(),
                    "eval_rate_eps": eps,
                    "mutation_rate_mps": eps * 0.8,
                    "inflight_requests": int(eps * 2),
                    "concurrency_limit": 64,
                    "queue_ready": q_ready,
                    "queue_mutation": int(q_ready / 2),
                    "queue_replay": random.randint(0, 5),
                    "straggler_count": random.randint(0, 2),
                    "latency_p50": lat,
                    "latency_p95": lat * 2.5,
                    "error_rate": 0.01 if random.random() > 0.9 else 0.0,
                    "run_id": RUN_ID,
                    "island_id": island
                }
                
                path = TELEMETRY_DIR / f"telemetry_{RUN_ID}_{island}.json"
                with open(path, "w") as f:
                    json.dump(snap, f)
            
            time.sleep(0.25)
            
    except KeyboardInterrupt:
        print("\n🛑 Simulation stopped.")

def main():
    setup_dirs()
    nodes, edges, timeline = generate_graph()
    write_evolution_file(nodes, edges, timeline)
    simulate_telemetry_loop()

if __name__ == "__main__":
    main()
