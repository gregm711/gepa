import os
import glob
import json
import asyncio
from pathlib import Path
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import uvicorn

print(f"Starting server setup... CWD: {os.getcwd()}")
print(f"Assets exists: {os.path.exists('assets')}")

app = FastAPI(title="TurboGEPA Live")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount assets directory
app.mount("/assets", StaticFiles(directory="assets"), name="assets")

# Configuration
TURBO_DIR = Path(".turbo_gepa")
EVO_DIR = TURBO_DIR / "evolution"
TELEMETRY_DIR = TURBO_DIR / "telemetry"

def get_latest_run_id() -> Optional[str]:
    """Find the most recently modified evolution file."""
    try:
        # Check current.json first
        current_ptr = EVO_DIR / "current.json"
        if current_ptr.exists():
            try:
                data = json.loads(current_ptr.read_text())
                rid = data.get("run_id")
                # Strip island suffix to get base run ID (e.g. "xyz-island0" -> "xyz")
                if rid and "-island" in rid:
                    return rid.split("-island")[0]
                return rid
            except:
                pass
        
        # Fallback to latest modified json file
        files = list(EVO_DIR.glob("*.json"))
        files = [f for f in files if f.name not in ("current.json", "current_summary.json")]
        if not files:
            return None
        latest = max(files, key=os.path.getmtime)
        # Strip island suffix if present
        stem = latest.stem
        if "-island" in stem:
            return stem.split("-island")[0]
        return stem
    except Exception:
        return None

@app.get("/api/status")
async def get_status():
    run_id = get_latest_run_id()
    return {
        "status": "online",
        "active_run_id": run_id
    }

def _merge_evolution_data(base_id: str, files: List[Path]) -> Dict[str, Any]:
    """Merge multiple island evolution snapshots into one."""
    merged = {
        "run_id": base_id,
        "evolution_stats": {
            "mutations_requested": 0,
            "mutations_generated": 0,
            "mutations_enqueued": 0,
            "mutations_promoted": 0,
            "unique_parents": 0,
            "unique_children": 0,
            "evolution_edges": 0,
            "total_evaluations": 0,
        },
        "lineage": [],
        "run_metadata": {
            "best_quality": 0.0,
            "evaluations": 0,
        },
        "metrics": {},
        "timeline": [] 
    }
    
    best_quality_so_far = -1.0
    
    for f in files:
        try:
            data = json.loads(f.read_text())
            
            # Merge lineage
            # We assume fingerprints are unique or identical across islands (which they are)
            merged["lineage"].extend(data.get("lineage", []))
            
            # Merge stats
            stats = data.get("evolution_stats", {})
            for k in ["mutations_requested", "mutations_generated", "mutations_enqueued", 
                      "mutations_promoted", "total_evaluations"]:
                merged["evolution_stats"][k] += stats.get(k, 0)
            
            merged["evolution_stats"]["unique_parents"] += stats.get("unique_parents", 0)
            merged["evolution_stats"]["unique_children"] += stats.get("unique_children", 0)
            merged["evolution_stats"]["evolution_edges"] += stats.get("evolution_edges", 0)

            # Merge Metadata (Max/Sum)
            meta = data.get("run_metadata", {})
            merged["run_metadata"]["evaluations"] += meta.get("evaluations", 0)
            
            q = meta.get("best_quality", 0.0)
            if q > best_quality_so_far:
                best_quality_so_far = q
                merged["run_metadata"]["best_quality"] = q
                merged["run_metadata"]["best_prompt"] = meta.get("best_prompt")
                merged["run_metadata"]["best_quality_shard"] = meta.get("best_quality_shard")
                merged["metrics"] = data.get("metrics", {})
                
            if data.get("timeline"):
                merged["timeline"].extend(data["timeline"])

        except Exception:
            pass
            
    # Sort timeline by evaluations
    if merged["timeline"]:
        merged["timeline"].sort(key=lambda x: x.get("evaluations", 0))
        
    return merged

@app.get("/api/telemetry/{run_id}")
async def get_telemetry(run_id: str):
    """Merge operational telemetry with evolution stats."""
    try:
        # Check if this is a multi-island run
        # Look for files starting with run_id + "-island"
        island_files = list(EVO_DIR.glob(f"{run_id}-island*.json"))
        single_file = EVO_DIR / f"{run_id}.json"
        
        evo_data = {}
        telemetry_data = []
        
        if single_file.exists():
            # Single file mode
            try:
                evo_data = json.loads(single_file.read_text())
            except:
                pass
            # Telemetry for single file
            telemetry_files = list(TELEMETRY_DIR.glob(f"telemetry_{run_id}_*.json"))
            for tf in telemetry_files:
                try:
                    telemetry_data.append(json.loads(tf.read_text()))
                except:
                    pass
                    
        elif island_files:
            # Multi-island mode
            evo_data = _merge_evolution_data(run_id, island_files)
            
            # Telemetry for all islands
            # Glob pattern: telemetry_{run_id}-island*
            telemetry_files = list(TELEMETRY_DIR.glob(f"telemetry_{run_id}-island*.json"))
            for tf in telemetry_files:
                try:
                    telemetry_data.append(json.loads(tf.read_text()))
                except:
                    pass
        
        return {
            "telemetry": telemetry_data,
            "evolution": evo_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Serve the frontend
@app.get("/")
async def serve_index():
    return FileResponse("scripts/viz/index.html")

if __name__ == "__main__":
    print("🚀 TurboGEPA Live Server starting on http://localhost:8082")
    uvicorn.run(app, host="0.0.0.0", port=8082, log_level="error")
