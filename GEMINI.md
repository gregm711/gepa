# TurboGEPA Project Context

## Project Overview
TurboGEPA is a high-performance, asynchronous fork of the GEPA (Genetic-Pareto) framework, designed for **high-throughput prompt evolution**. It optimizes LLM prompts by trading token efficiency for maximum speed using:
- **Async/Await Orchestration:** Maximizes concurrency using `asyncio` and `httpx`.
- **Island-Based Parallelism:** Runs multiple evolutionary "islands" concurrently to maintain diversity.
- **ASHA Successive Halving:** Prunes underperforming candidates early (e.g., evaluating on 5% -> 20% -> 100% of the dataset).
- **Dual Mutation Strategy:** Combines reflective edits (fixing failures) with spec induction (generating fresh instructions).

## Key Technologies & Stack
- **Language:** Python 3.10+
- **Package Manager:** `uv` (recommended), `pip` / `setuptools`
- **LLM Interface:** `litellm`
- **Testing:** `pytest` (with `pytest-asyncio`)
- **Linting/Formatting:** `ruff`
- **Core Dependencies:** `httpx`, `xxhash`, `dspy-ai` (optional), `pydantic` (implied)

## Development Setup

### Installation
The project uses `uv` for dependency management.
```bash
# Install dependencies
uv sync --extra dev --python 3.11

# Activate virtual environment
source .venv/bin/activate
```

### Building
```bash
# Build the package
uv build
```

## Running Tests
Tests are located in the `tests/` directory and use `pytest`.
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/turbo_gepa/test_evaluator_verification.py
```

## Linting & Formatting
The project uses `ruff` for both linting and formatting.
```bash
# Check for linting errors
ruff check .

# Format code
ruff format .
```

## Key Directories & Files
- **`src/turbo_gepa/`**: Core source code.
  - `adapters/`: Integrations (e.g., `DefaultAdapter` for simple prompts, `DSpyAdapter` for DSPy).
  - `distributed/`: Code for running distributed workers.
  - `strategies/`: Mutation strategies (`incremental_reflection`, `spec_induction`).
  - `config.py`: Configuration classes (`Config`).
  - `evaluator.py`: Core evaluation logic (ASHA).
  - `orchestrator.py`: Manages the optimization loop.
- **`examples/`**: Usage examples (e.g., `quickstart.py`, `aime_benchmark.py`).
- **`tests/`**: Unit and integration tests.
- **`scripts/`**: Analysis and visualization tools.
- **`pyproject.toml`**: Project configuration and dependencies.
- **`.turbo_gepa/`**: Default directory for runtime artifacts (logs, cache, metrics).

## Common Usage Patterns
**Running a Quickstart:**
```bash
python examples/quickstart.py
```

**Running a Benchmark:**
```bash
python examples/aime_benchmark_v2.py --mode turbo --dataset-size 30
```

**Distributed Worker:**
```bash
turbo-gepa-worker --help
```

## Architecture Notes
- **Config:** Heavy reliance on `turbo_gepa.config.Config` for tuning concurrency, shards, and strategies.
- **Concurrency:** `eval_concurrency` controls the number of inflight requests per island.
- **Caching:** Results are cached in `.turbo_gepa/cache` (using `xxhash` of inputs) to avoid redundant LLM calls.
- **Migration:** Islands exchange "elite" candidates periodically to cross-pollinate successful prompts.
