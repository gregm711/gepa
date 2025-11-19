"""
Turbo-Top: Real-time operational dashboard for TurboGEPA.

Usage:
    python scripts/turbo_top.py [run_id]

Run in a separate terminal pane to monitor engine internals.
"""

import sys
import time
import json
import glob
import os
from datetime import datetime
from pathlib import Path
from typing import List

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn

console = Console()

def load_telemetry(log_dir: str = ".turbo_gepa/telemetry") -> List[dict]:
    """Find and load all active telemetry files."""
    files = glob.glob(os.path.join(log_dir, "telemetry_*.json"))
    snapshots = []
    for f in files:
        try:
            with open(f, "r") as handle:
                data = json.load(handle)
                snapshots.append(data)
        except Exception:
            pass
    return snapshots

def make_layout() -> Layout:
    layout = Layout(name="root")
    layout.split(
        Layout(name="header", size=3),
        Layout(name="metrics", size=12),
        Layout(name="queues", size=8),
        Layout(name="footer", size=3),
    )
    return layout

def render_metrics(snapshots: List[dict]) -> Panel:
    if not snapshots:
        return Panel("Waiting for telemetry...", title="Metrics")
        
    # Aggregate if multiple islands
    total_eps = sum(s.get("eval_rate_eps", 0) for s in snapshots)
    total_mps = sum(s.get("mutation_rate_mps", 0) for s in snapshots)
    total_inflight = sum(s.get("inflight_requests", 0) for s in snapshots)
    total_limit = sum(s.get("concurrency_limit", 1) for s in snapshots)
    
    # Weighted average for latency
    avg_p95 = max(s.get("latency_p95", 0) for s in snapshots) if snapshots else 0
    avg_p50 = max(s.get("latency_p50", 0) for s in snapshots) if snapshots else 0
    
    table = Table(expand=True, box=None)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")
    table.add_column("Visual", style="green")
    
    # EPS
    table.add_row(
        "Throughput (Evals/s)", 
        f"{total_eps:.1f}", 
        "█" * min(20, int(total_eps))
    )
    
    # Inflight
    util = total_inflight / max(1, total_limit)
    color = "green" if util < 0.9 else "red"
    table.add_row(
        "Inflight Requests",
        f"{total_inflight} / {total_limit}",
        Text("█" * int(util * 20), style=color)
    )
    
    # Latency
    lat_color = "green" if avg_p95 < 5.0 else "yellow"
    if avg_p95 > 15.0: lat_color = "red"
    table.add_row(
        "Latency (p95)",
        f"{avg_p95:.2f}s",
        Text("█" * min(20, int(avg_p95 * 2)), style=lat_color)
    )
    
    return Panel(table, title="🔥 Engine Flow")

def render_queues(snapshots: List[dict]) -> Panel:
    if not snapshots:
        return Panel("Waiting...", title="Queues")
        
    table = Table(expand=True)
    table.add_column("Island")
    table.add_column("Ready Queue")
    table.add_column("Mutation Buffer")
    table.add_column("Stragglers")
    
    for s in snapshots:
        island = str(s.get("island_id", "?"))
        ready = s.get("queue_ready", 0)
        mut = s.get("queue_mutation", 0)
        strag = s.get("straggler_count", 0)
        
        ready_style = "white"
        if ready < 5: ready_style = "red blink" # Starvation warning
        
        mut_style = "white"
        if mut > 100: mut_style = "yellow" # Backlog warning
        
        table.add_row(
            island,
            Text(str(ready), style=ready_style),
            Text(str(mut), style=mut_style),
            str(strag)
        )
        
    return Panel(table, title="🌊 Queue Pressure")

def run_dashboard(run_id_filter: str = None):
    layout = make_layout()
    with Live(layout, refresh_per_second=4) as live:
        while True:
            snapshots = load_telemetry()
            if run_id_filter:
                snapshots = [s for s in snapshots if s.get("run_id") == run_id_filter]
            
            # Update layout parts
            ts = datetime.now().strftime("%H:%M:%S")
            status = "🟢 ONLINE" if snapshots else "⚪ WAITING"
            layout["header"].update(
                Panel(f"TurboGEPA Operational Dashboard | {ts} | {status}", style="bold white on blue")
            )
            layout["metrics"].update(render_metrics(snapshots))
            layout["queues"].update(render_queues(snapshots))
            
            time.sleep(0.25)

if __name__ == "__main__":
    rid = sys.argv[1] if len(sys.argv) > 1 else None
    try:
        run_dashboard(rid)
    except KeyboardInterrupt:
        console.print("\n[yellow]Dashboard closed.[/yellow]")
