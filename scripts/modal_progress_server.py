"""
Modal web service serving the TurboGEPA dashboard + evolution JSON from the shared volume.
"""
from __future__ import annotations

import json
import os
import glob
from pathlib import Path
from typing import Optional

import modal
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

REPO_ROOT = Path(__file__).resolve().parents[1]
VOLUME_MOUNT = Path("/mnt/turbogepa")
# Assuming the standard output location relative to the volume root
TURBO_DIR = VOLUME_MOUNT / ".turbo_gepa" 
EVO_DIR = TURBO_DIR / "evolution"
TELEMETRY_DIR = TURBO_DIR / "telemetry"

image = (
    modal.Image.debian_slim()
    .pip_install("fastapi[standard]")
    # Copy the viz frontend and assets
    .add_local_file(REPO_ROOT / "scripts" / "viz" / "index.html", "/app/scripts/viz/index.html", copy=True)
    .add_local_dir(REPO_ROOT / "assets", remote_path="/app/assets", copy=True)
)

app = modal.App("turbogepa-progress", image=image)
volume = modal.Volume.from_name("turbo-gepa-cache", create_if_missing=True)

fastapi_app = FastAPI(title="TurboGEPA Live (Modal)")

# Enable CORS
fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount assets directory
fastapi_app.mount("/assets", StaticFiles(directory="/app/assets"), name="assets")

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

@fastapi_app.get("/api/status")
async def get_status():
    try:
        volume.reload()
    except Exception:
        pass
        
    run_id = get_latest_run_id()
    return {
        "status": "online",
        "active_run_id": run_id,
        "environment": "modal"
    }

@fastapi_app.get("/api/telemetry/{run_id}")
async def get_telemetry(run_id: str):
    """Merge operational telemetry with evolution stats."""
    try:
        # Refresh volume to get latest writes
        try:
            volume.reload()
        except Exception:
            pass

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

@fastapi_app.get("/")
async def serve_dashboard():
    return FileResponse("/app/scripts/viz/index.html")

@app.function(volumes={str(VOLUME_MOUNT): volume})
@modal.asgi_app()
def progress_service():
    return fastapi_app