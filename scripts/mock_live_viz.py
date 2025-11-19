import json
import os
import time
import random
import uuid

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def main():
    # Setup paths
    base_dir = ".turbo_gepa/evolution"
    ensure_dir(base_dir)
    
    run_id = f"mock_run_{int(time.time())}"
    current_file = os.path.join(base_dir, "current.json")
    run_file = os.path.join(base_dir, f"{run_id}.json")

    # Point current run to this mock
    with open(current_file, "w") as f:
        json.dump({"run_id": run_id}, f)
    
    print(f"🚀 Starting mock simulation for run: {run_id}")
    print(f"📂 Watching file: {run_file}")
    print("👉 Run 'python scripts/viz_server.py' to see the dashboard!")

    # Simulation state
    evaluations = 0
    best_quality = 0.0
    cost = 0.0
    start_time = time.time()
    
    # Initial population
    lineage = []
    timeline = []
    
    # Islands configuration
    islands = [0, 1, 2]
    
    # Create a seed node
    seed_fp = uuid.uuid4().hex[:12]
    lineage.append({
        "fingerprint": seed_fp,
        "generation": 0,
        "quality": 0.15,
        "status": "promoted",
        "shard_fraction": 1.0,
        "prompt": "You are a helpful assistant.",
        "prompt_full": "You are a helpful assistant.",
        "origin_island": 0,
        "current_island": 0
    })
    
    parent_children = {seed_fp: []}
    active_parents = [seed_fp]
    
    # Loop to simulate progress
    for round_idx in range(1, 50):
        # 1. Generate new candidates (mutations)
        num_mutations = random.randint(3, 8)
        
        for _ in range(num_mutations):
            parent = random.choice(active_parents)
            child_fp = uuid.uuid4().hex[:12]
            
            # Simulate quality improvement with some noise
            parent_node = next(n for n in lineage if n["fingerprint"] == parent)
            base_quality = parent_node["quality"]
            
            # Determine island (sometimes migrate)
            island = parent_node["current_island"]
            migrated_from = None
            if random.random() < 0.1: # 10% chance to migrate
                new_island = random.choice([i for i in islands if i != island])
                migrated_from = island
                island = new_island
            
            # Improve or regress
            delta = random.uniform(-0.05, 0.08)
            quality = max(0.0, min(1.0, base_quality + delta))
            
            # Update global best
            if quality > best_quality:
                best_quality = quality
            
            # Shard progression logic (mock)
            shard = 0.05 if random.random() < 0.5 else (0.2 if random.random() < 0.5 else 1.0)
            
            status = "promoted" if quality > base_quality else "pruned"
            if status == "promoted":
                active_parents.append(child_fp)
                if len(active_parents) > 5: # Keep active pool small
                    active_parents.pop(0)
                parent_children.setdefault(parent, []).append(child_fp)
                parent_children[child_fp] = []

            # Cost accumulation (random per eval)
            step_cost = random.uniform(0.001, 0.005)
            cost += step_cost
            evaluations += 1
            
            lineage.append({
                "fingerprint": child_fp,
                "generation": parent_node["generation"] + 1,
                "quality": quality,
                "status": status,
                "shard_fraction": shard,
                "prompt": f"Variation of {parent[:6]}...",
                "prompt_full": f"Full prompt text for {child_fp}...",
                "origin_island": parent_node["origin_island"], # Origin stays same
                "current_island": island,
                "migrated_from_island": migrated_from
            })

        # 2. Update snapshot
        elapsed = time.time() - start_time
        
        snapshot = {
            "run_id": run_id,
            "metrics": {
                "total_cost_usd": cost,
                "evaluations_total": evaluations,
            },
            "run_metadata": {
                "run_id": run_id,
                "evaluations": evaluations,
                "best_quality": best_quality,
                "best_quality_shard": 1.0,
                "best_prompt": "Mock best prompt..."
            },
            "evolution_stats": {
                "parent_children": parent_children,
                "unique_parents": len(parent_children),
                "evolution_edges": sum(len(v) for v in parent_children.values()),
            },
            "lineage": lineage,
            "timeline": timeline
        }
        
        # Add timeline point
        timeline.append({
            "evaluations": evaluations,
            "best_quality": best_quality,
            "elapsed": elapsed
        })
        
        # Write atomically
        tmp_file = run_file + ".tmp"
        with open(tmp_file, "w") as f:
            json.dump(snapshot, f)
        os.replace(tmp_file, run_file)
        
        print(f"Round {round_idx}: Evals={evaluations}, Best={best_quality:.3f}, Cost=${cost:.3f}")
        time.sleep(1.0) # Sleep 1s between updates to allow UI to poll

if __name__ == "__main__":
    main()
