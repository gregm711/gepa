import os
import glob
import json
import asyncio
from pathlib import Path
from typing import Optional

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
                return data.get("run_id")
            except:
                pass
        
        # Fallback to latest modified json file
        files = list(EVO_DIR.glob("*.json"))
        files = [f for f in files if f.name not in ("current.json", "current_summary.json")]
        if not files:
            return None
        latest = max(files, key=os.path.getmtime)
        return latest.stem
    except Exception:
        return None

@app.get("/api/status")
async def get_status():
    run_id = get_latest_run_id()
    return {
        "status": "online",
        "active_run_id": run_id
    }

@app.get("/api/telemetry/{run_id}")
async def get_telemetry(run_id: str):
    """Merge operational telemetry with evolution stats."""
    try:
        # 1. Get high-frequency telemetry
        telemetry_files = list(TELEMETRY_DIR.glob(f"telemetry_{run_id}_*.json"))
        telemetry_data = []
        for tf in telemetry_files:
            try:
                telemetry_data.append(json.loads(tf.read_text()))
            except:
                pass
        
        # 2. Get evolution snapshot (slower update)
        evo_file = EVO_DIR / f"{run_id}.json"
        evo_data = {}
        if evo_file.exists():
            try:
                evo_data = json.loads(evo_file.read_text())
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
    print("🚀 TurboGEPA Live Server starting on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="error")