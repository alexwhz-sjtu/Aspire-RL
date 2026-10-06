"""FastAPI control plane for ASPIRE online-RL episodes."""

from __future__ import annotations

import hmac
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated
from uuid import UUID

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import Response
from pydantic import ValidationError

from aspire.sim.online_rl.artifacts import ArtifactStore
from aspire.sim.online_rl.broker.scheduler import ReplayScheduler
from aspire.sim.online_rl.broker.state import EpisodeService
from aspire.sim.online_rl.broker.store import EpisodeStore
from aspire.sim.online_rl.contracts import (
    PROTOCOL_VERSION,
    ActionRequest,
    ActionResponse,
    CloseEpisodeRequest,
    CreateEpisodeRequest,
    EpisodeCreated,
    FindingsRequest,
    HealthResponse,
    TerminalScoreRequest,
    TerminalScoreResponse,
)
from aspire.sim.online_rl.executor import ReplayExecutor
from aspire.sim.online_rl.runner.worker import SpawnReplayExecutor
from aspire.sim.online_rl.task_registry import TaskRegistry


@dataclass(frozen=True)
class BrokerSettings:
    campaign_manifest: Path
    state_db: Path
    artifact_root: Path
    api_reference_root: Path
    admin_token: str
    token_secret: str
    max_simulator_concurrency: int = 2
    cleanup_orphans_older_than_seconds: int | None = None


def _bearer_token(authorization: str | None) -> str:
    prefix = "Bearer "
    if authorization is None or not authorization.startswith(prefix):
        raise HTTPException(status_code=401, detail="Bearer token required")
    return authorization[len(prefix) :]


def create_app(
    settings: BrokerSettings, *, executor: ReplayExecutor | None = None
) -> FastAPI:
    registry = TaskRegistry(settings.campaign_manifest)
    api_reference_root = registry.verify_api_snapshot_path(
        settings.api_reference_root
    )
    store = EpisodeStore(settings.state_db)
    if settings.cleanup_orphans_older_than_seconds is not None:
        store.fail_orphaned_episodes(
            stale_after_seconds=settings.cleanup_orphans_older_than_seconds
        )
    artifact_store = ArtifactStore(settings.artifact_root)
    scheduler = ReplayScheduler(
        executor or SpawnReplayExecutor(),
        max_concurrency=settings.max_simulator_concurrency,
    )
    service = EpisodeService(
        registry=registry,
        store=store,
        artifacts=artifact_store,
        scheduler=scheduler,
        token_secret=settings.token_secret,
        api_reference_root=api_reference_root,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        store.close()

    app = FastAPI(
        title="ASPIRE Online-RL Broker",
        version="1",
        lifespan=lifespan,
    )
    app.state.service = service

    def require_admin(
        authorization: Annotated[str | None, Header()] = None,
    ) -> None:
        if not hmac.compare_digest(
            _bearer_token(authorization), settings.admin_token
        ):
            raise HTTPException(status_code=401, detail="invalid admin token")

    def require_episode(
        episode_id: UUID,
        authorization: Annotated[str | None, Header()] = None,
    ) -> str:
        token = _bearer_token(authorization)
        try:
            service.authorize(str(episode_id), token)
        except PermissionError as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc
        return str(episode_id)

    @app.get("/health/live", response_model=HealthResponse)
    async def health_live() -> HealthResponse:
        return HealthResponse(status="ok", protocol_version=PROTOCOL_VERSION)

    @app.get("/health/ready", response_model=HealthResponse)
    async def health_ready() -> HealthResponse:
        if not settings.api_reference_root.is_dir():
            return HealthResponse(
                status="not_ready",
                protocol_version=PROTOCOL_VERSION,
                detail="API reference root does not exist",
            )
        try:
            runner_ready = await scheduler.ready()
        except Exception as exc:
            return HealthResponse(
                status="not_ready",
                protocol_version=PROTOCOL_VERSION,
                detail=f"simulator runner health check failed: {exc}",
            )
        if not runner_ready:
            return HealthResponse(
                status="not_ready",
                protocol_version=PROTOCOL_VERSION,
                detail="one or more simulator runners are not ready",
            )
        return HealthResponse(status="ok", protocol_version=PROTOCOL_VERSION)

    @app.post(
        "/v1/episodes",
        response_model=EpisodeCreated,
        dependencies=[Depends(require_admin)],
    )
    async def create_episode(request: CreateEpisodeRequest) -> EpisodeCreated:
        try:
            return await service.create_episode(request)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post(
        "/v1/episodes/{episode_id}/actions", response_model=ActionResponse
    )
    async def execute_action(
        request: dict[str, object],
        authorized_episode_id: str = Depends(require_episode),
    ) -> ActionResponse:
        try:
            try:
                validated = ActionRequest.model_validate(request)
            except ValidationError as exc:
                try:
                    request_id = str(UUID(str(request["request_id"])))
                    expected_state_version = int(request["expected_state_version"])
                except (KeyError, TypeError, ValueError) as envelope_exc:
                    raise HTTPException(
                        status_code=422,
                        detail="request_id and expected_state_version must be valid",
                    ) from envelope_exc
                return await service.record_policy_error(
                    authorized_episode_id,
                    request_id=request_id,
                    expected_state_version=expected_state_version,
                    code="invalid_tool_arguments",
                    message=str(exc),
                    request_payload=request,
                )
            return await service.execute_action(authorized_episode_id, validated)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RuntimeError as exc:
            if "state version conflict" in str(exc) or "concurrent" in str(exc):
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            raise

    @app.get(
        "/v1/admin/metrics",
        dependencies=[Depends(require_admin)],
    )
    async def get_metrics() -> dict[str, object]:
        return store.metrics_snapshot()

    @app.get(
        "/v1/admin/episodes/{episode_id}/frozen-code",
        dependencies=[Depends(require_admin)],
    )
    async def get_frozen_code(episode_id: UUID) -> Response:
        try:
            code_sha256, code = service.read_frozen_code(str(episode_id))
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return Response(
            content=code.encode(),
            media_type="text/x-python; charset=utf-8",
            headers={
                "ETag": code_sha256,
                "X-Code-SHA256": code_sha256,
                "Content-Disposition": 'attachment; filename="fix_code.py"',
            },
        )

    @app.get(
        "/v1/admin/episodes/{episode_id}",
        dependencies=[Depends(require_admin)],
    )
    async def get_episode_audit(episode_id: UUID) -> dict[str, object]:
        try:
            record = store.get_episode(str(episode_id))
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        terminal_score = store.get_terminal_score(str(episode_id))
        return {
            "episode_id": record.episode_id,
            "task_uid": record.task_uid,
            "state": record.state,
            "state_version": record.state_version,
            "counters": record.counters.model_dump(mode="json"),
            "current_code_sha256": record.current_code_sha256,
            "frozen_code_sha256": record.frozen_code_sha256,
            "findings": record.findings,
            "terminal_score": terminal_score,
            "created_at": record.created_at,
            "updated_at": record.updated_at,
        }

    @app.get("/v1/episodes/{episode_id}/artifacts/{artifact_id}")
    async def get_artifact(
        artifact_id: str,
        authorized_episode_id: str = Depends(require_episode),
    ) -> Response:
        try:
            ref, data = service.read_artifact(authorized_episode_id, artifact_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return Response(
            content=data,
            media_type=ref.media_type,
            headers={"ETag": ref.sha256, "X-Artifact-Kind": ref.kind},
        )

    @app.post("/v1/episodes/{episode_id}/findings")
    async def save_findings(
        request: FindingsRequest,
        authorized_episode_id: str = Depends(require_episode),
    ) -> dict[str, object]:
        record = await service.save_findings(authorized_episode_id, request)
        return {"state": record.state, "state_version": record.state_version}

    @app.post(
        "/v1/episodes/{episode_id}/terminal-score",
        response_model=TerminalScoreResponse,
    )
    async def terminal_score(
        request: TerminalScoreRequest,
        authorized_episode_id: str = Depends(require_episode),
    ) -> TerminalScoreResponse:
        return await service.terminal_score(authorized_episode_id, request)

    @app.post("/v1/episodes/{episode_id}/close")
    async def close_episode(
        request: CloseEpisodeRequest,
        authorized_episode_id: str = Depends(require_episode),
    ) -> dict[str, object]:
        record = await service.close_episode(authorized_episode_id, request)
        return {"state": record.state, "state_version": record.state_version}

    return app
