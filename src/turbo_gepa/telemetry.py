"""
Real-time telemetry data structures for operational monitoring.
"""

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class TelemetrySnapshot:
    """Operational snapshot of the engine's internal state."""

    # Timestamp
    timestamp: float = field(default_factory=time.time)

    # Flow Rates (per second)
    eval_rate_eps: float = 0.0
    mutation_rate_mps: float = 0.0

    # Saturation & Backpressure
    inflight_requests: int = 0
    concurrency_limit: int = 1
    semaphore_utilization: float = 0.0

    # Queue Depths
    queue_ready: int = 0
    queue_mutation: int = 0
    queue_replay: int = 0
    straggler_count: int = 0

    # Latency Stats (Rolling window)
    latency_p50: float = 0.0
    latency_p95: float = 0.0

    # Provider Health
    error_rate: float = 0.0
    throttle_events: int = 0

    # System State
    run_id: str = "unknown"
    island_id: int = 0
    status: str = "running" # running, paused, stopping, etc.
    total_cost_usd: float = 0.0


class TelemetryCollector:
    """Thread-safe singleton for collecting and publishing telemetry."""

    _instance: Optional["TelemetryCollector"] = None

    def __init__(self, run_id: str, island_id: int = 0, log_dir: str = ".turbo_gepa/telemetry"):
        self.run_id = run_id
        self.island_id = island_id
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.telemetry_file = self.log_dir / f"telemetry_{run_id}_{island_id}.json"

        # Internal counters
        self._evals_completed = 0
        self._mutations_generated = 0
        self._errors = 0
        self._start_time = time.time()
        self._last_flush = time.time()
        self._last_eval_count = 0
        self._last_mutation_count = 0
        self._eval_rate_ema = 0.0
        self._mutation_rate_ema = 0.0

        # Latency tracking
        self._latencies: list[float] = []

    @classmethod
    def get_instance(cls) -> Optional["TelemetryCollector"]:
        return cls._instance

    @classmethod
    def initialize(cls, run_id: str, island_id: int = 0) -> "TelemetryCollector":
        cls._instance = cls(run_id, island_id)
        return cls._instance

    def record_eval_completion(self, latency: float, error: bool = False):
        self._evals_completed += 1
        self._latencies.append(latency)
        if len(self._latencies) > 20: # Keep window small for responsiveness
            self._latencies.pop(0)
        if error:
            self._errors += 1

    def record_mutation_generated(self):
        self._mutations_generated += 1

    def snapshot(self,
                 inflight: int,
                 limit: int,
                 queue_ready: int,
                 queue_mutation: int,
                 queue_replay: int,
                 straggler_count: int,
                 cost: float = 0.0) -> TelemetrySnapshot:

        now = time.time()
        delta = max(0.1, now - self._last_flush)

        # Calculate rates with EMA smoothing
        raw_eval_rate = (self._evals_completed - self._last_eval_count) / delta
        raw_mut_rate = (self._mutations_generated - self._last_mutation_count) / delta
        
        self._eval_rate_ema = 0.2 * raw_eval_rate + 0.8 * self._eval_rate_ema
        self._mutation_rate_ema = 0.2 * raw_mut_rate + 0.8 * self._mutation_rate_ema

        # Calculate latency stats
        if self._latencies:
            sorted_lat = sorted(self._latencies)
            p50 = sorted_lat[int(len(sorted_lat) * 0.5)]
            p95 = sorted_lat[int(len(sorted_lat) * 0.95)]
        else:
            p50 = 0.0
            p95 = 0.0

        # Error rate (last window)
        # Simplified for now: total errors / total evals
        err_rate = self._errors / max(1, self._evals_completed)

        snap = TelemetrySnapshot(
            timestamp=now,
            eval_rate_eps=self._eval_rate_ema,
            mutation_rate_mps=self._mutation_rate_ema,
            inflight_requests=inflight,
            concurrency_limit=limit,
            semaphore_utilization=inflight / max(1, limit),
            queue_ready=queue_ready,
            queue_mutation=queue_mutation,
            queue_replay=queue_replay,
            straggler_count=straggler_count,
            latency_p50=p50,
            latency_p95=p95,
            error_rate=err_rate,
            run_id=self.run_id,
            island_id=self.island_id,
            total_cost_usd=cost
        )

        # Update baselines for next rate calc
        self._last_flush = now
        self._last_eval_count = self._evals_completed
        self._last_mutation_count = self._mutations_generated

        return snap

    def publish(self, snapshot: TelemetrySnapshot):
        """Atomic write of telemetry snapshot."""
        try:
            temp_path = self.telemetry_file.with_suffix(".tmp")
            with open(temp_path, "w") as f:
                json.dump(asdict(snapshot), f)
            temp_path.replace(self.telemetry_file)
        except Exception:
            pass # Never crash the engine on telemetry write failure
