"""Immutable, resumable fixed-code evaluation with no model calls."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from aspire.sim.online_rl.artifacts import ArtifactStore
from aspire.sim.online_rl.contracts import HELDOUT_SEEDS
from aspire.sim.online_rl.executor import ExecutionArtifact, ReplayExecutor
from aspire.sim.online_rl.reward import cap_rl_seed_reward
from aspire.sim.online_rl.runner.worker import SpawnReplayExecutor
from aspire.sim.online_rl.snapshots import canonical_json_digest, sha256_file
from aspire.sim.online_rl.task_registry import TaskRegistry


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()  # noqa: UP017 - Python 3.10


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def evaluate_frozen_code(
    *,
    campaign_manifest: str | Path,
    task_uid: str,
    code_path: str | Path,
    output_root: str | Path,
    resume: bool,
    seeds: tuple[int, ...] = HELDOUT_SEEDS,
    executor: ReplayExecutor | None = None,
    infrastructure_retries: int = 3,
) -> Path:
    if tuple(sorted(set(seeds))) != seeds or any(seed not in HELDOUT_SEEDS for seed in seeds):
        raise ValueError("held-out seeds must be a sorted unique subset of 1..50")
    if not 0 <= infrastructure_retries <= 10:
        raise ValueError("infrastructure_retries must be between 0 and 10")
    campaign_path = Path(campaign_manifest).resolve()
    campaign_digest = sha256_file(campaign_path)
    registry = TaskRegistry(campaign_path)
    task = registry.get(task_uid)
    task = task.model_copy(
        update={
            "env_config": str(registry.resolve_env_config(task)),
            "skill_snapshot": str(registry.resolve_skill_snapshot(task)),
        }
    )
    code_file = Path(code_path).resolve()
    code_sha256 = sha256_file(code_file)
    identity = {
        "protocol_version": registry.manifest.protocol_version,
        "campaign_id": registry.manifest.campaign_id,
        "campaign_manifest_digest": campaign_digest,
        "task_uid": task_uid,
        "code_sha256": code_sha256,
        "env_config_digest": task.env_config_digest,
        "skill_snapshot_digest": task.skill_snapshot_digest,
        "seeds": list(seeds),
    }
    run_id = canonical_json_digest(identity)[:16]
    run_root = Path(output_root).resolve() / task.suite / task.task_name / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    manifest_path = run_root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("identity") != identity:
            raise RuntimeError("held-out manifest identity mismatch")
        if not resume:
            raise FileExistsError("held-out run exists; pass resume=True")
    else:
        manifest = {
            "schema_version": 1,
            "identity": identity,
            "status": "running",
            "model_calls": 0,
            "results": {},
            "created_at": _now(),
            "updated_at": _now(),
        }
        _write_json_atomic(manifest_path, manifest)
    manifest.setdefault("infrastructure_retry_events", [])
    manifest["infrastructure_retry_limit"] = infrastructure_retries

    code = code_file.read_text()
    artifact_store = ArtifactStore(run_root / "artifacts")
    replay_executor = executor or SpawnReplayExecutor()
    for seed in seeds:
        key = str(seed)
        if key in manifest["results"]:
            continue
        if sha256_file(code_file) != code_sha256:
            raise RuntimeError("frozen code changed during held-out evaluation")
        for attempt in range(infrastructure_retries + 1):
            try:
                execution = replay_executor.replay(
                    task,
                    code,
                    seed,
                    timeout_seconds=task.budgets.replay_timeout_seconds,
                )
                break
            except Exception as exc:
                manifest["infrastructure_retry_events"].append(
                    {
                        "seed": seed,
                        "attempt": attempt + 1,
                        "error_type": type(exc).__name__,
                        "message": str(exc)[:2000],
                        "recorded_at": _now(),
                    }
                )
                manifest["updated_at"] = _now()
                _write_json_atomic(manifest_path, manifest)
                if attempt >= infrastructure_retries:
                    raise
        artifact_refs = []
        execution_artifacts = [
            *execution.artifacts,
            ExecutionArtifact("stdout", "text/plain; charset=utf-8", execution.stdout.encode()),
            ExecutionArtifact("stderr", "text/plain; charset=utf-8", execution.stderr.encode()),
        ]
        for artifact in execution_artifacts:
            artifact_refs.append(
                artifact_store.put(
                    artifact_id=str(uuid4()),
                    kind=artifact.kind,
                    media_type=artifact.media_type,
                    data=artifact.data,
                ).model_dump(mode="json")
            )
        manifest["results"][key] = {
            "seed": seed,
            "env_reward": execution.env_reward,
            "sandbox_rc": execution.sandbox_rc,
            "task_completed": execution.task_completed,
            "cap_rl_reward": cap_rl_seed_reward(execution.env_reward, execution.sandbox_rc),
            "trace_summary": execution.trace_summary,
            "artifacts": artifact_refs,
            "recorded_at": _now(),
        }
        manifest["updated_at"] = _now()
        _write_json_atomic(manifest_path, manifest)

    rows = list(manifest["results"].values())
    manifest["status"] = "complete" if len(rows) == len(seeds) else "partial"
    manifest["trials"] = len(rows)
    manifest["success_rate"] = (
        sum(bool(row["task_completed"]) for row in rows) / len(rows) if rows else 0.0
    )
    manifest["cap_rl_mean"] = (
        sum(float(row["cap_rl_reward"]) for row in rows) / len(rows) if rows else 0.0
    )
    manifest["updated_at"] = _now()
    if sha256_file(code_file) != code_sha256:
        raise RuntimeError("frozen code changed during held-out evaluation")
    if sha256_file(campaign_path) != campaign_digest:
        raise RuntimeError("campaign manifest changed during held-out evaluation")
    registry.verify_task_sources(registry.get(task_uid))
    _write_json_atomic(manifest_path, manifest)
    return manifest_path
