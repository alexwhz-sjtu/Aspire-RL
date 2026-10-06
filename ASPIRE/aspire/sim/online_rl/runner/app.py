"""Internal authenticated simulator runner service."""

from __future__ import annotations

import asyncio
import base64
import hmac
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, Header, HTTPException

from aspire.sim.online_rl.contracts import CampaignTask
from aspire.sim.online_rl.runner.worker import SpawnReplayExecutor
from aspire.sim.online_rl.snapshots import sha256_bytes


@dataclass(frozen=True)
class RunnerSettings:
    token: str


def create_runner_app(settings: RunnerSettings) -> FastAPI:
    if not settings.token:
        raise ValueError("runner token must not be empty")
    executor = SpawnReplayExecutor()
    app = FastAPI(title="ASPIRE Online-RL Runner", version="1")

    def authorize(authorization: Annotated[str | None, Header()] = None) -> None:
        if authorization is None or not hmac.compare_digest(
            authorization, f"Bearer {settings.token}"
        ):
            raise HTTPException(status_code=401, detail="invalid runner token")

    def task_with_local_config(body: dict[str, Any], root: Path) -> CampaignTask:
        task = CampaignTask.model_validate(body["task"])
        config_text = body.get("env_config_text")
        if not isinstance(config_text, str):
            raise HTTPException(status_code=422, detail="env_config_text is required")
        if sha256_bytes(config_text.encode()) != task.env_config_digest:
            raise HTTPException(status_code=422, detail="environment config digest mismatch")
        config_path = root / "environment.yaml"
        config_path.write_text(config_text)
        return task.model_copy(update={"env_config": str(config_path)})

    def serialize_artifacts(artifacts: list[Any]) -> list[dict[str, str]]:
        return [
            {
                "kind": artifact.kind,
                "media_type": artifact.media_type,
                "data_base64": base64.b64encode(artifact.data).decode(),
            }
            for artifact in artifacts
        ]

    @app.get("/health/ready")
    async def health_ready() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/internal/v1/observe")
    async def observe(
        body: dict[str, Any],
        authorization: Annotated[str | None, Header()] = None,
    ) -> dict[str, Any]:
        authorize(authorization)
        with tempfile.TemporaryDirectory(prefix="aspire-runner-config-") as temp_dir:
            task = task_with_local_config(body, Path(temp_dir))
            result = await asyncio.to_thread(
                executor.observe,
                task,
                timeout_seconds=task.budgets.replay_timeout_seconds,
            )
        return {
            "task_language": result.task_language,
            "text": result.text,
            "artifacts": serialize_artifacts(result.artifacts),
        }

    @app.post("/internal/v1/replay")
    async def replay(
        body: dict[str, Any],
        authorization: Annotated[str | None, Header()] = None,
    ) -> dict[str, Any]:
        authorize(authorization)
        code = body.get("code")
        seed = body.get("seed")
        if not isinstance(code, str) or not isinstance(seed, int):
            raise HTTPException(status_code=422, detail="code and integer seed are required")
        with tempfile.TemporaryDirectory(prefix="aspire-runner-config-") as temp_dir:
            task = task_with_local_config(body, Path(temp_dir))
            result = await asyncio.to_thread(
                executor.replay,
                task,
                code,
                seed,
                timeout_seconds=task.budgets.replay_timeout_seconds,
            )
        return {
            "seed": result.seed,
            "env_reward": result.env_reward,
            "sandbox_rc": result.sandbox_rc,
            "task_completed": result.task_completed,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "trace_summary": result.trace_summary,
            "artifacts": serialize_artifacts(result.artifacts),
        }

    return app
