import asyncio

import pytest

from turbo_gepa.utils.llm_request_manager import LLMRequestManager


class Fake429(Exception):
    status_code = 429

    def __init__(self, msg: str = "429") -> None:
        super().__init__(msg)


class Fake500(Exception):
    status_code = 500

    def __init__(self, msg: str = "500") -> None:
        super().__init__(msg)


@pytest.mark.asyncio
async def test_throttle_reduces_effective_concurrency():
    manager = LLMRequestManager(
        max_concurrency=3,
        base_backoff=0.01,
        max_backoff=0.02,
        relax_window=0.05,
        jitter_max=0.0,
    )
    attempts = {"count": 0}

    async def coro():
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise Fake429("rate limited")
        return "ok"

    result = await manager.run("throttle-test", coro, max_attempts=2, base_delay=0.01)

    assert result == "ok"
    assert manager.throttle_events == 1
    assert manager.effective_concurrency < manager.max_concurrency


@pytest.mark.asyncio
async def test_effective_concurrency_recovers_after_quiet_window():
    manager = LLMRequestManager(
        max_concurrency=3,
        base_backoff=0.01,
        max_backoff=0.02,
        relax_window=0.02,
        jitter_max=0.0,
    )
    attempts = {"count": 0}

    async def coro():
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise Fake500("server error")
        return "ok"

    # Trigger throttle then allow relax
    await manager.run("throttle-test", coro, max_attempts=2, base_delay=0.01)
    down = manager.effective_concurrency
    assert down < manager.max_concurrency

    await asyncio.sleep(0.03)
    await manager.run("recovery-test", lambda: asyncio.sleep(0), max_attempts=1)

    assert manager.effective_concurrency == manager.max_concurrency
