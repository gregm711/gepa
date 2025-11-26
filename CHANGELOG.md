# Changelog

All notable changes to this project will be documented in this file.

## [0.1.1] - 2025-11-26

### Added
- Live dashboard now supports multi-island runs with merged evolution timelines and telemetry.
- Post-run Markdown report generator for summarizing orchestrator results.
- Live viz mock generator simulation mode for quicker demos.

### Fixed
- Resolved dashboard pathing for distributed runs and ensured telemetry aggregates across islands.
- Improved reporting stability (string literal fix) and added startup logging for easier debugging.
- Smoothed telemetry metrics to reduce noisy spikes in live views.

## [0.1.0] - 2025-11-19

### Added
- **TurboGEPA Core**: High-throughput, async prompt evolution framework.
- **Island-Based Parallelism**: Concurrent optimization islands with elite migration.
- **ASHA Pruning**: Early stopping of underperforming candidates to save compute.
- **Dual Mutation Strategy**: `incremental_reflection` and `spec_induction` (Prompt-MII).
- **DSpy Adapter**: Support for optimizing DSPy program instructions (`DSpyAdapter`).
- **Telemetry**: Real-time operational monitoring (`turbo_top.py`, `viz_server.py`).

### Fixed
- **Critical**: Removed duplicate and conflicting `get_candidate_lineage_data` method in orchestrator.
- **Critical**: Restored missing `run_id` property in `Orchestrator` preventing distributed runs.
- **Critical**: Fixed `NameError` in orchestrator streaming logic (`decision` variable).
- **Metrics**: Consolidated duplicate metric definitions (`baseline_quality`, `target_quality`).
- **DSpy Integration**: Fixed `ImportError` for `LogLevel` and removed deprecated config attributes.
- **Tests**: Removed outdated legacy tests (`test_final_rung_controller.py`, deprecated gating tests).
- **Linting**: Applied comprehensive `ruff` formatting and fixes across the codebase.
