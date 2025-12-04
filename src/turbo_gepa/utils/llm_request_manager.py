"""Shared LiteLLM request manager with status-aware throttling/backoff."""

from __future__ import annotations

import asyncio
import logging
import random
import time
from typing import Awaitable, Callable

try:
    from turbo_gepa.logging.logger import LogLevel  # type: ignore
except Exception:  # pragma: no cover - optional import for generic reuse
    LogLevel = None  # type: ignore


class LLMRequestManager:
    """Coordinated throttling and retries across all LiteLLM calls."""

    def __init__(
        self,
        max_concurrency: int,
        *,
        logger: object | None = None,
        base_backoff: float = 1.5,
        max_backoff: float = 8.0,
        relax_window: float = 5.0,
        jitter_max: float = 0.2,
    ) -> None:
        self._max_concurrency = max(1, max_concurrency)
        self._semaphore = asyncio.Semaphore(self._max_concurrency)
        self._effective_concurrency = self._max_concurrency
        self._inflight = 0
        self._inflight_lock = asyncio.Lock()
        self._backoff_until: float = 0.0
        self._base_backoff = base_backoff
        self._max_backoff = max_backoff
        self._relax_window = relax_window
        self._jitter_max = max(0.0, jitter_max)
        self._throttle_events = 0
        self._logger = logger
        self._last_throttle: float = 0.0
        self._last_backoff_length: float = 0.0

    @property
    def throttle_events(self) -> int:
        return self._throttle_events

    @property
    def effective_concurrency(self) -> int:
        return self._effective_concurrency

    @property
    def max_concurrency(self) -> int:
        return self._max_concurrency

    @property
    def backoff_remaining(self) -> float:
        return max(0.0, self._backoff_until - time.time())

    def update_logger(self, logger: object | None) -> None:
        if logger is not None:
            self._logger = logger

    async def run(
        self,
        label: str,
        coro_factory: Callable[[], Awaitable[object]],
        *,
        max_attempts: int = 3,
        base_delay: float | None = None,
    ):
        """Execute a request with shared backoff on 429/5xx errors."""
        base = base_delay if base_delay is not None else self._base_backoff
        last_exc: Exception | None = None

        for attempt in range(1, max_attempts + 1):
            await self._respect_backoff()
            try:
                async with self._inflight_slot():
                    async with self._semaphore:
                        result = await coro_factory()
                    self._maybe_relax()
                    return result
            except Exception as exc:
                last_exc = exc
                transient, delay_hint = self._classify_transient(exc)
                if not transient or attempt >= max_attempts:
                    raise
                delay = delay_hint if delay_hint is not None else base * (1.6 ** (attempt - 1))
                delay = min(self._max_backoff, delay)
                if self._jitter_max:
                    delay += random.uniform(0.0, self._jitter_max)
                self._set_backoff(delay)
                self._shrink_effective()
                self._warn(label, exc, delay)
                await asyncio.sleep(min(delay, 1.0))

        if last_exc:
            raise last_exc

    def _inflight_slot(self):
        # Lightweight context manager to honor effective_concurrency.
        class _Slot:
            def __init__(self, mgr: LLMRequestManager) -> None:
                self.mgr = mgr

            async def __aenter__(self):
                mgr = self.mgr
                while True:
                    async with mgr._inflight_lock:
                        if mgr._inflight < mgr._effective_concurrency:
                            mgr._inflight += 1
                            return
                    await asyncio.sleep(0.01)

            async def __aexit__(self, exc_type, exc, tb):
                mgr = self.mgr
                async with mgr._inflight_lock:
                    mgr._inflight = max(0, mgr._inflight - 1)

        return _Slot(self)

    async def _respect_backoff(self) -> None:
        remaining = self._backoff_until - time.time()
        if remaining > 0:
            await asyncio.sleep(remaining)

    def _set_backoff(self, delay: float) -> None:
        now = time.time()
        self._backoff_until = max(self._backoff_until, now + max(0.0, delay))
        self._throttle_events += 1
        self._last_throttle = now
        self._last_backoff_length = max(self._last_backoff_length, delay)

    def _warn(self, label: str, exc: Exception, delay: float) -> None:
        msg = f"🚦 LLM throttle for '{label}' after {type(exc).__name__}: {exc} | backoff {delay:.1f}s"
        logger = self._logger
        try:
            if logger is not None and hasattr(logger, "log") and LogLevel is not None:
                logger.log(msg, LogLevel.WARNING)  # type: ignore[attr-defined]
            else:
                logging.warning(msg)
        except Exception:
            pass

    def _classify_transient(self, exc: Exception) -> tuple[bool, float | None]:
        status = self._extract_status(exc)
        if status is None:
            status = self._status_from_message(str(exc))

        if status == 429:
            retry_after = self._extract_retry_after(exc)
            return True, retry_after
        if status in (500, 502, 503, 504):
            return True, None
        return False, None

    def _shrink_effective(self) -> None:
        new_eff = max(1, self._effective_concurrency - 1)
        self._effective_concurrency = new_eff

    def _maybe_relax(self) -> None:
        # Gradually return to max after a quiet window
        now = time.time()
        quiet = now - self._last_throttle
        min_quiet = max(self._relax_window, self._last_backoff_length + self._relax_window)
        if quiet >= min_quiet and self._effective_concurrency < self._max_concurrency:
            self._effective_concurrency += 1
            if self._effective_concurrency > self._max_concurrency:
                self._effective_concurrency = self._max_concurrency

    def _extract_status(self, exc: Exception) -> int | None:
        for attr in ("status_code", "status", "code"):
            val = getattr(exc, attr, None)
            if isinstance(val, int):
                return val
        resp = getattr(exc, "response", None)
        if resp is not None:
            for attr in ("status_code", "status"):
                val = getattr(resp, attr, None)
                if isinstance(val, int):
                    return val
            headers = getattr(resp, "headers", None)
            if headers and isinstance(headers, dict):
                retry_after = headers.get("Retry-After") or headers.get("retry-after")
                if retry_after:
                    return 429
        return None

    def _extract_retry_after(self, exc: Exception) -> float | None:
        resp = getattr(exc, "response", None)
        headers = getattr(resp, "headers", None)
        retry_after = None
        if headers and isinstance(headers, dict):
            retry_after = headers.get("Retry-After") or headers.get("retry-after")
        if retry_after is None:
            return None
        try:
            value = float(retry_after)
            if value > 0:
                return value
        except Exception:
            return None
        return None

    def _status_from_message(self, message: str) -> int | None:
        lower = message.lower()
        if "429" in lower or "too many requests" in lower or "rate limit" in lower:
            return 429
        if "internal server error" in lower or "cloudflare" in lower or "5xx" in lower:
            return 500
        if "502" in lower or "bad gateway" in lower:
            return 502
        if "503" in lower or "service unavailable" in lower or "openrouterexception" in lower:
            return 503
        if "504" in lower or "gateway timeout" in lower:
            return 504
        if "timeout" in lower or "timed out" in lower:
            return 504
        if "connection reset" in lower or "connection aborted" in lower:
            return 503
        return None


_GLOBAL_MANAGER: LLMRequestManager | None = None


def get_llm_request_manager(max_concurrency: int, *, logger: object | None = None) -> LLMRequestManager:
    """Return a shared manager sized to the provided concurrency."""
    global _GLOBAL_MANAGER
    desired = max(1, max_concurrency)
    if _GLOBAL_MANAGER is None:
        _GLOBAL_MANAGER = LLMRequestManager(desired, logger=logger)
    elif desired > _GLOBAL_MANAGER._max_concurrency:
        _GLOBAL_MANAGER = LLMRequestManager(desired, logger=logger)
    else:
        _GLOBAL_MANAGER.update_logger(logger)
    return _GLOBAL_MANAGER
