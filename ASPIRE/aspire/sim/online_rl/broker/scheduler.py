"""Bounded async scheduling around blocking simulator executors."""

from __future__ import annotations

import asyncio

from aspire.sim.online_rl.contracts import CampaignTask
from aspire.sim.online_rl.executor import (
    InitialObservationExecution,
    ReplayExecution,
    ReplayExecutor,
)


class ReplayScheduler:
    def __init__(self, executor: ReplayExecutor, max_concurrency: int = 2) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be positive")
        self.executor = executor
        self._semaphore = asyncio.Semaphore(max_concurrency)

    async def ready(self) -> bool:
        healthcheck = getattr(self.executor, "healthcheck", None)
        if healthcheck is None:
            return True
        return bool(await asyncio.to_thread(healthcheck, timeout_seconds=10))

    async def observe(
        self, task: CampaignTask, *, timeout_seconds: int
    ) -> InitialObservationExecution:
        async with self._semaphore:
            return await asyncio.to_thread(
                self.executor.observe, task, timeout_seconds=timeout_seconds
            )

    async def replay(
        self,
        task: CampaignTask,
        code: str,
        seed: int,
        *,
        timeout_seconds: int,
    ) -> ReplayExecution:
        async with self._semaphore:
            return await asyncio.to_thread(
                self.executor.replay,
                task,
                code,
                seed,
                timeout_seconds=timeout_seconds,
            )
