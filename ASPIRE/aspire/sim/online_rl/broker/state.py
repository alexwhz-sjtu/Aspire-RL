"""Authoritative episode state machine and tool dispatch."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from aspire.sim.online_rl.artifacts import ArtifactStore
from aspire.sim.online_rl.broker.scheduler import ReplayScheduler
from aspire.sim.online_rl.broker.store import EpisodeRecord, EpisodeStore
from aspire.sim.online_rl.contracts import (
    ActionRequest,
    ActionResponse,
    ArtifactRef,
    CampaignTask,
    CloseEpisodeRequest,
    CreateEpisodeRequest,
    EpisodeCounters,
    EpisodeCreated,
    EpisodeState,
    FinalSubmissionAction,
    FindingsRequest,
    Observation,
    PolicyError,
    ReadSearchAction,
    ReplayAction,
    SeedReplayResult,
    TerminalScoreRequest,
    TerminalScoreResponse,
    TraceInspectionAction,
    WriteEditAction,
)
from aspire.sim.online_rl.executor import ExecutionArtifact
from aspire.sim.online_rl.reward import cap_rl_episode_reward
from aspire.sim.online_rl.runner.isolation import UnsafeProgramError, validate_program
from aspire.sim.online_rl.snapshots import sha256_bytes
from aspire.sim.online_rl.task_registry import TaskRegistry


class PolicyActionError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def apply_unified_patch(source: str, patch: str) -> str:
    """Apply a strict single-file unified diff without invoking external tools."""
    source_lines = source.splitlines(keepends=True)
    patch_lines = patch.splitlines(keepends=True)
    result: list[str] = []
    source_index = 0
    patch_index = 0
    hunk_pattern = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")

    while patch_index < len(patch_lines) and patch_lines[patch_index].startswith(
        ("--- ", "+++ ")
    ):
        patch_index += 1
    if patch_index >= len(patch_lines):
        raise PolicyActionError("invalid_patch", "unified patch contains no hunks")

    while patch_index < len(patch_lines):
        match = hunk_pattern.match(patch_lines[patch_index])
        if match is None:
            raise PolicyActionError(
                "invalid_patch", f"expected hunk header: {patch_lines[patch_index].rstrip()}"
            )
        old_start = int(match.group(1)) - 1
        old_count = int(match.group(2) or "1")
        if old_start < source_index or old_start > len(source_lines):
            raise PolicyActionError("invalid_patch", "hunk source position is invalid")
        result.extend(source_lines[source_index:old_start])
        source_index = old_start
        patch_index += 1
        consumed = 0

        while patch_index < len(patch_lines) and not patch_lines[patch_index].startswith("@@ "):
            line = patch_lines[patch_index]
            patch_index += 1
            if line.startswith("\\ No newline at end of file"):
                continue
            if not line or line[0] not in {" ", "+", "-"}:
                raise PolicyActionError("invalid_patch", "invalid unified patch line")
            prefix, content = line[0], line[1:]
            if prefix in {" ", "-"}:
                if source_index >= len(source_lines) or source_lines[source_index] != content:
                    raise PolicyActionError("patch_conflict", "patch context does not match code")
                source_index += 1
                consumed += 1
            if prefix in {" ", "+"}:
                result.append(content)
        if consumed != old_count:
            raise PolicyActionError(
                "invalid_patch",
                f"hunk declared {old_count} source lines but consumed {consumed}",
            )

    result.extend(source_lines[source_index:])
    return "".join(result)


class EpisodeService:
    def __init__(
        self,
        *,
        registry: TaskRegistry,
        store: EpisodeStore,
        artifacts: ArtifactStore,
        scheduler: ReplayScheduler,
        token_secret: str,
        api_reference_root: str | Path,
    ) -> None:
        if not token_secret:
            raise ValueError("token_secret must not be empty")
        self.registry = registry
        self.store = store
        self.artifacts = artifacts
        self.scheduler = scheduler
        self._token_secret = token_secret.encode()
        self.api_reference_root = Path(api_reference_root).resolve()
        self._locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    def _task(self, record: EpisodeRecord) -> CampaignTask:
        task = self.registry.get(record.task_uid)
        return task.model_copy(
            update={
                "env_config": str(self.registry.resolve_env_config(task)),
                "skill_snapshot": str(self.registry.resolve_skill_snapshot(task)),
            }
        )

    def token_for(self, episode_id: str) -> str:
        digest = hmac.new(
            self._token_secret, episode_id.encode(), hashlib.sha256
        ).digest()
        return base64.urlsafe_b64encode(digest).decode().rstrip("=")

    def authorize(self, episode_id: str, token: str) -> None:
        expected = self.token_for(episode_id)
        if not hmac.compare_digest(expected, token):
            raise PermissionError("invalid episode token")

    def _ensure_within_deadline(self, record: EpisodeRecord) -> None:
        created_at = datetime.fromisoformat(record.created_at)
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)  # noqa: UP017
        elapsed = (
            datetime.now(timezone.utc)  # noqa: UP017 - Python 3.10
            - created_at
        ).total_seconds()
        timeout = self._task(record).budgets.episode_timeout_seconds
        if elapsed >= timeout:
            raise PolicyActionError(
                "episode_timeout",
                f"episode wall-time budget of {timeout} seconds is exhausted",
            )

    async def create_episode(self, request: CreateEpisodeRequest) -> EpisodeCreated:
        existing = self.store.find_by_client_episode_id(str(request.client_episode_id))
        if existing is not None:
            task = self._task(existing)
            self.registry.verify_task_sources(
                self.registry.get(existing.task_uid)
            )
            return EpisodeCreated(
                episode_id=UUID(existing.episode_id),
                scoped_token=self.token_for(existing.episode_id),
                state=existing.state,
                state_version=existing.state_version,
                initial_observation=existing.observation,
                skill_snapshot_digest=task.skill_snapshot_digest,
                api_snapshot_digest=self.registry.manifest.api_snapshot_digest,
                actor_prompt_digest=self.registry.manifest.actor_prompt_digest,
                tool_schema_digest=self.registry.manifest.tool_schema_digest,
                processor_digest=self.registry.manifest.processor_digest,
                protocol_digest=task.protocol_digest,
                budgets=task.budgets,
            )

        task = self.registry.get(request.task_uid)
        self.registry.verify_task_sources(task)
        task = task.model_copy(
            update={
                "env_config": str(self.registry.resolve_env_config(task)),
                "skill_snapshot": str(self.registry.resolve_skill_snapshot(task)),
            }
        )
        observation_execution = await self.scheduler.observe(
            task, timeout_seconds=task.budgets.replay_timeout_seconds
        )
        episode_id = str(uuid4())
        refs: list[ArtifactRef] = []
        for artifact in observation_execution.artifacts:
            artifact_ref = self.artifacts.put(
                artifact_id=str(uuid4()),
                kind=artifact.kind,
                media_type=artifact.media_type,
                data=artifact.data,
            )
            refs.append(artifact_ref)
        observation = Observation(
            task_language=observation_execution.task_language,
            text=observation_execution.text,
            artifacts=refs,
        )
        record = self.store.create_episode(
            episode_id=episode_id,
            client_episode_id=str(request.client_episode_id),
            task_uid=task.task_uid,
            observation=observation,
        )
        for artifact_ref in refs:
            self.store.add_artifact(episode_id, artifact_ref)
        return EpisodeCreated(
            episode_id=UUID(episode_id),
            scoped_token=self.token_for(episode_id),
            state=record.state,
            state_version=record.state_version,
            initial_observation=observation,
            skill_snapshot_digest=task.skill_snapshot_digest,
            api_snapshot_digest=self.registry.manifest.api_snapshot_digest,
            actor_prompt_digest=self.registry.manifest.actor_prompt_digest,
            tool_schema_digest=self.registry.manifest.tool_schema_digest,
            processor_digest=self.registry.manifest.processor_digest,
            protocol_digest=task.protocol_digest,
            budgets=task.budgets,
        )

    async def execute_action(
        self, episode_id: str, request: ActionRequest
    ) -> ActionResponse:
        async with self._locks[episode_id]:
            cached = self.store.get_action(episode_id, str(request.request_id))
            if cached is not None:
                return cached
            record = self.store.get_episode(episode_id)
            if record.state_version != request.expected_state_version:
                raise RuntimeError(
                    f"state version conflict: expected {request.expected_state_version}, "
                    f"current {record.state_version}"
                )
            try:
                self._ensure_within_deadline(record)
                result, state, counters, current_sha, frozen_sha = await self._dispatch(
                    record, request.action
                )
                updated = self.store.update_episode(
                    episode_id,
                    expected_version=record.state_version,
                    state=state,
                    counters=counters,
                    current_code_sha256=current_sha,
                    frozen_code_sha256=frozen_sha,
                )
                response = ActionResponse(
                    ok=True,
                    state=updated.state,
                    state_version=updated.state_version,
                    result=result,
                    counters=updated.counters,
                )
            except (PolicyActionError, UnsafeProgramError) as exc:
                updated = self.store.update_episode(
                    episode_id,
                    expected_version=record.state_version,
                )
                response = ActionResponse(
                    ok=False,
                    state=updated.state,
                    state_version=updated.state_version,
                    error=PolicyError(
                        code=getattr(exc, "code", "unsafe_program"), message=str(exc)
                    ),
                    counters=updated.counters,
                )
            except Exception as exc:
                self.store.record_event(
                    episode_id,
                    event_class="infrastructure",
                    code=type(exc).__name__,
                    message=str(exc),
                )
                self.store.update_episode(
                    episode_id,
                    expected_version=record.state_version,
                    state=EpisodeState.INFRA_FAILED,
                )
                raise
            self.store.record_action(
                episode_id,
                str(request.request_id),
                request.model_dump(mode="json"),
                response,
            )
            return response

    async def record_policy_error(
        self,
        episode_id: str,
        *,
        request_id: str,
        expected_state_version: int,
        code: str,
        message: str,
        request_payload: dict[str, Any],
    ) -> ActionResponse:
        """Durably record malformed model actions that fail contract validation."""
        async with self._locks[episode_id]:
            cached = self.store.get_action(episode_id, request_id)
            if cached is not None:
                return cached
            record = self.store.get_episode(episode_id)
            if record.state_version != expected_state_version:
                raise RuntimeError(
                    f"state version conflict: expected {expected_state_version}, "
                    f"current {record.state_version}"
                )
            try:
                self._ensure_within_deadline(record)
            except PolicyActionError as exc:
                code = exc.code
                message = str(exc)
            updated = self.store.update_episode(
                episode_id, expected_version=record.state_version
            )
            response = ActionResponse(
                ok=False,
                state=updated.state,
                state_version=updated.state_version,
                error=PolicyError(code=code, message=message[:4000]),
                counters=updated.counters,
            )
            self.store.record_action(
                episode_id, request_id, request_payload, response
            )
            return response

    async def _dispatch(
        self, record: EpisodeRecord, action: Any
    ) -> tuple[
        dict[str, Any], EpisodeState, EpisodeCounters, str | None, str | None
    ]:
        if isinstance(action, ReadSearchAction):
            result = self._read_search(record, action)
            return result, record.state, record.counters, None, None
        if isinstance(action, WriteEditAction):
            return self._write_edit(record, action)
        if isinstance(action, ReplayAction):
            return await self._replay(record, action)
        if isinstance(action, TraceInspectionAction):
            result = self._trace_inspection(record, action)
            return result, record.state, record.counters, None, None
        if isinstance(action, FinalSubmissionAction):
            return self._final_submission(record, action)
        raise PolicyActionError("unknown_action", f"unsupported action: {type(action).__name__}")

    def _resolve_read_root(self, record: EpisodeRecord, root_name: str) -> Path | None:
        if root_name == "skills":
            return Path(self._task(record).skill_snapshot).resolve()
        if root_name == "api_reference":
            return self.api_reference_root
        return None

    def _read_search(self, record: EpisodeRecord, action: ReadSearchAction) -> dict[str, Any]:
        if record.state in {EpisodeState.COMPLETED, EpisodeState.CLOSED}:
            raise PolicyActionError("episode_closed", "episode is no longer readable")
        if action.root == "workspace":
            if action.path not in {None, "current_code.py", "initial_code.py", "fix_code.py"}:
                raise PolicyActionError("path_not_allowed", "workspace exposes only robot code")
            if record.current_code_sha256 is None:
                return {"matches": [], "text": "", "truncated": False}
            text = self.store.get_code(record.episode_id, record.current_code_sha256)
            if action.pattern:
                try:
                    pattern = re.compile(action.pattern)
                except re.error as exc:
                    raise PolicyActionError("invalid_pattern", str(exc)) from exc
                lines = [line for line in text.splitlines() if pattern.search(line)]
                text = "\n".join(lines)
            encoded = text.encode()[: action.max_bytes]
            return {
                "path": "current_code.py",
                "text": encoded.decode(errors="replace"),
                "truncated": len(text.encode()) > action.max_bytes,
                "code_sha256": record.current_code_sha256,
            }

        root = self._resolve_read_root(record, action.root)
        assert root is not None
        relative = Path(action.path or ".")
        if relative.is_absolute() or ".." in relative.parts:
            raise PolicyActionError("path_not_allowed", "absolute paths and '..' are forbidden")
        target = (root / relative).resolve()
        if target != root and root not in target.parents:
            raise PolicyActionError("path_not_allowed", "path escapes approved root")
        candidates = [target] if target.is_file() else sorted(
            path for path in target.rglob("*") if path.is_file() and not path.is_symlink()
        )
        chunks: list[str] = []
        remaining = action.max_bytes
        try:
            pattern = re.compile(action.pattern) if action.pattern else None
        except re.error as exc:
            raise PolicyActionError("invalid_pattern", str(exc)) from exc
        for path in candidates:
            text = path.read_text(errors="replace")
            if pattern is not None:
                text = "\n".join(
                    line for line in text.splitlines() if pattern.search(line)
                )
            if not text:
                continue
            header = f"\n--- {path.relative_to(root).as_posix()} ---\n"
            chunk = (header + text).encode()[:remaining]
            chunks.append(chunk.decode(errors="replace"))
            remaining -= len(chunk)
            if remaining <= 0:
                break
        return {"text": "".join(chunks), "truncated": remaining <= 0}

    def _write_edit(
        self, record: EpisodeRecord, action: WriteEditAction
    ) -> tuple[dict[str, Any], EpisodeState, EpisodeCounters, str, None]:
        if record.state not in {
            EpisodeState.CONTEXT_READY,
            EpisodeState.INITIAL_CODE_WRITTEN,
            EpisodeState.DEBUGGING,
        }:
            raise PolicyActionError("code_frozen", "robot code can no longer be edited")
        if action.mode == "replace":
            code = action.content
            if record.current_code_sha256 is not None and action.base_sha256 != record.current_code_sha256:
                raise PolicyActionError("code_conflict", "base_sha256 must match current code")
        else:
            if record.current_code_sha256 is None:
                raise PolicyActionError("missing_base", "cannot patch before initial code exists")
            if action.base_sha256 != record.current_code_sha256:
                raise PolicyActionError("code_conflict", "base_sha256 must match current code")
            source = self.store.get_code(record.episode_id, record.current_code_sha256)
            code = apply_unified_patch(source, action.content)
        if not code.strip():
            raise PolicyActionError("empty_code", "robot program must not be empty")
        validate_program(code)
        digest = sha256_bytes(code.encode())
        revision = self.store.add_code_revision(record.episode_id, digest, code)
        counters = record.counters.model_copy(
            update={"code_revisions": max(record.counters.code_revisions, revision)}
        )
        state = (
            EpisodeState.INITIAL_CODE_WRITTEN
            if record.current_code_sha256 is None
            else EpisodeState.DEBUGGING
        )
        return {"code_sha256": digest, "revision": revision}, state, counters, digest, None

    async def _replay(
        self, record: EpisodeRecord, action: ReplayAction
    ) -> tuple[dict[str, Any], EpisodeState, EpisodeCounters, None, None]:
        if record.state not in {EpisodeState.INITIAL_CODE_WRITTEN, EpisodeState.DEBUGGING}:
            raise PolicyActionError("replay_not_allowed", "replay is not allowed in this state")
        if action.code_sha256 != record.current_code_sha256:
            raise PolicyActionError("code_conflict", "replay code must be the current revision")
        if not record.counters.baseline_replays and action.seeds[0] != 51:
            raise PolicyActionError("seed_order", "the first replay seed must be 51")

        baseline = dict(record.counters.baseline_replays)
        repairs = dict(record.counters.repair_replays)
        task = self._task(record)
        for seed in action.seeds:
            if baseline.get(seed, 0) == 0:
                baseline[seed] = 1
            elif repairs.get(seed, 0) >= task.budgets.max_repair_replays_per_seed:
                raise PolicyActionError(
                    "replay_budget", f"seed {seed} exhausted its repair replay budget"
                )
            else:
                repairs[seed] = repairs.get(seed, 0) + 1

        code = self.store.get_code(record.episode_id, action.code_sha256)
        results: list[dict[str, Any]] = []
        for seed in action.seeds:
            execution = await self.scheduler.replay(
                task,
                code,
                seed,
                timeout_seconds=task.budgets.replay_timeout_seconds,
            )
            result = self._record_execution(
                record.episode_id,
                action.code_sha256,
                execution,
                terminal=False,
            )
            results.append(result.model_dump(mode="json"))
        counters = record.counters.model_copy(
            update={"baseline_replays": baseline, "repair_replays": repairs}
        )
        return {"seeds": results}, EpisodeState.DEBUGGING, counters, None, None

    def _record_execution(
        self,
        episode_id: str,
        code_sha256: str,
        execution: Any,
        *,
        terminal: bool,
    ) -> SeedReplayResult:
        replay_id = str(uuid4())
        refs: list[ArtifactRef] = []
        artifact_payloads = list(execution.artifacts)
        artifact_payloads.extend(
            [
                ExecutionArtifact(
                    kind="stdout",
                    media_type="text/plain; charset=utf-8",
                    data=execution.stdout.encode(),
                ),
                ExecutionArtifact(
                    kind="stderr",
                    media_type="text/plain; charset=utf-8",
                    data=execution.stderr.encode(),
                ),
            ]
        )
        for payload in artifact_payloads:
            ref = self.artifacts.put(
                artifact_id=str(uuid4()),
                kind=payload.kind,
                media_type=payload.media_type,
                data=payload.data,
            )
            refs.append(ref)
            self.store.add_artifact(episode_id, ref, replay_id=replay_id)
        result = SeedReplayResult(
            replay_id=UUID(replay_id),
            seed=execution.seed,
            code_sha256=code_sha256,
            env_reward=execution.env_reward,
            sandbox_rc=execution.sandbox_rc,
            task_completed=execution.task_completed,
            trace_summary=execution.trace_summary,
            artifacts=refs,
        )
        self.store.add_replay(
            replay_id=replay_id,
            episode_id=episode_id,
            seed=execution.seed,
            code_sha256=code_sha256,
            result=result.model_dump(mode="json"),
            terminal=terminal,
        )
        return result

    def _trace_inspection(
        self, record: EpisodeRecord, action: TraceInspectionAction
    ) -> dict[str, Any]:
        selected_refs: list[ArtifactRef] = []
        replay_summaries = []
        text_bytes = 0
        image_count = 0
        selector_kinds = {
            "trace": {"trace"},
            "stdout": {"stdout"},
            "stderr": {"stderr"},
            "rgb_keyframes": {"rgb_keyframe"},
            "mask": {"mask"},
            "grasp": {"grasp"},
            "ik": {"trace"},
        }
        allowed_kinds = set().union(*(selector_kinds[value] for value in action.selectors))
        texts: list[dict[str, str]] = []
        for replay_uuid in action.replay_ids:
            replay_id = str(replay_uuid)
            try:
                replay_summaries.append(
                    self.store.get_replay(record.episode_id, replay_id)
                )
            except KeyError as exc:
                raise PolicyActionError("unknown_replay", str(exc)) from exc
            for ref in self.store.list_replay_artifacts(record.episode_id, replay_id):
                if ref.kind not in allowed_kinds:
                    continue
                if ref.media_type.startswith("image/"):
                    if image_count >= action.max_images:
                        continue
                    image_count += 1
                    selected_refs.append(ref)
                elif ref.media_type.startswith("text/") or ref.media_type == "application/json":
                    remaining = action.max_bytes - text_bytes
                    if remaining <= 0:
                        continue
                    data = self.artifacts.read(ref.sha256)[:remaining]
                    text_bytes += len(data)
                    texts.append(
                        {
                            "artifact_id": ref.artifact_id,
                            "kind": ref.kind,
                            "text": data.decode(errors="replace"),
                        }
                    )
                else:
                    selected_refs.append(ref)
        return {
            "replays": replay_summaries,
            "texts": texts,
            "artifacts": [ref.model_dump(mode="json") for ref in selected_refs],
            "truncated": text_bytes >= action.max_bytes,
        }

    def _final_submission(
        self, record: EpisodeRecord, action: FinalSubmissionAction
    ) -> tuple[dict[str, Any], EpisodeState, EpisodeCounters, None, str]:
        if record.state not in {EpisodeState.INITIAL_CODE_WRITTEN, EpisodeState.DEBUGGING}:
            raise PolicyActionError("submission_not_allowed", "code cannot be submitted now")
        if action.code_sha256 != record.current_code_sha256:
            raise PolicyActionError("code_conflict", "submission must freeze current code")
        missing = sorted(set(self._task(record).development_seeds) - set(record.counters.baseline_replays))
        if missing:
            raise PolicyActionError(
                "baseline_incomplete", f"all development seeds require a baseline replay: {missing}"
            )
        return (
            {"code_sha256": action.code_sha256},
            EpisodeState.CODE_FROZEN,
            record.counters,
            None,
            action.code_sha256,
        )

    async def save_findings(self, episode_id: str, request: FindingsRequest) -> EpisodeRecord:
        async with self._locks[episode_id]:
            record = self.store.get_episode(episode_id)
            if record.findings is not None:
                if record.findings != request.findings:
                    raise RuntimeError("findings are immutable once stored")
                return record
            if record.state != EpisodeState.CODE_FROZEN:
                raise RuntimeError("findings require frozen code")
            return self.store.update_episode(
                episode_id,
                expected_version=record.state_version,
                state=EpisodeState.REPORTING,
                findings=request.findings,
                set_findings=True,
            )

    async def terminal_score(
        self, episode_id: str, request: TerminalScoreRequest
    ) -> TerminalScoreResponse:
        async with self._locks[episode_id]:
            cached = self.store.get_terminal_score(episode_id)
            if cached is not None:
                return TerminalScoreResponse.model_validate(cached)
            record = self.store.get_episode(episode_id)
            frozen_sha = record.frozen_code_sha256
            if frozen_sha is None and request.auto_freeze:
                frozen_sha = record.current_code_sha256
                if frozen_sha is None:
                    response = TerminalScoreResponse(
                        code_sha256="", reward=0.0, seeds=[]
                    )
                    self.store.complete_terminal_score(
                        episode_id,
                        expected_version=record.state_version,
                        state=EpisodeState.POLICY_FAILED,
                        response=response.model_dump(mode="json"),
                    )
                    return response
                record = self.store.update_episode(
                    episode_id,
                    expected_version=record.state_version,
                    state=EpisodeState.CODE_FROZEN,
                    frozen_code_sha256=frozen_sha,
                )
                if request.fallback_findings is not None:
                    record = self.store.update_episode(
                        episode_id,
                        expected_version=record.state_version,
                        state=EpisodeState.REPORTING,
                        findings=request.fallback_findings,
                        set_findings=True,
                    )
            if frozen_sha is None or record.state not in {
                EpisodeState.CODE_FROZEN,
                EpisodeState.REPORTING,
            }:
                raise RuntimeError("terminal scoring requires frozen code")
            record = self.store.update_episode(
                episode_id,
                expected_version=record.state_version,
                state=EpisodeState.TERMINAL_SCORING,
            )
            task = self._task(record)
            code = self.store.get_code(episode_id, frozen_sha)
            seed_results: list[SeedReplayResult] = []
            try:
                for seed in task.terminal_seeds:
                    last_error: Exception | None = None
                    for _attempt in range(2):
                        try:
                            execution = await self.scheduler.replay(
                                task,
                                code,
                                seed,
                                timeout_seconds=task.budgets.replay_timeout_seconds,
                            )
                            break
                        except Exception as exc:
                            last_error = exc
                    else:
                        assert last_error is not None
                        raise last_error
                    seed_results.append(
                        self._record_execution(
                            episode_id, frozen_sha, execution, terminal=True
                        )
                    )
                reward = cap_rl_episode_reward(
                    (result.env_reward, result.sandbox_rc)
                    for result in seed_results
                )
                response = TerminalScoreResponse(
                    code_sha256=frozen_sha, reward=reward, seeds=seed_results
                )
            except Exception as exc:
                self.store.record_event(
                    episode_id,
                    event_class="infrastructure",
                    code=type(exc).__name__,
                    message=f"terminal scoring failed: {exc}",
                )
                self.store.update_episode(
                    episode_id,
                    expected_version=record.state_version,
                    state=EpisodeState.INFRA_FAILED,
                )
                raise
            self.store.complete_terminal_score(
                episode_id,
                expected_version=record.state_version,
                state=EpisodeState.COMPLETED,
                response=response.model_dump(mode="json"),
            )
            return response

    def read_artifact(self, episode_id: str, artifact_id: str) -> tuple[ArtifactRef, bytes]:
        ref = self.store.get_artifact(episode_id, artifact_id)
        return ref, self.artifacts.read(ref.sha256)

    def read_frozen_code(self, episode_id: str) -> tuple[str, str]:
        record = self.store.get_episode(episode_id)
        if record.frozen_code_sha256 is None:
            raise RuntimeError("episode has no frozen code")
        return (
            record.frozen_code_sha256,
            self.store.get_code(episode_id, record.frozen_code_sha256),
        )

    async def close_episode(
        self, episode_id: str, request: CloseEpisodeRequest
    ) -> EpisodeRecord:
        del request
        async with self._locks[episode_id]:
            record = self.store.get_episode(episode_id)
            if record.state == EpisodeState.CLOSED:
                return record
            return self.store.update_episode(
                episode_id,
                expected_version=record.state_version,
                state=EpisodeState.CLOSED,
            )
