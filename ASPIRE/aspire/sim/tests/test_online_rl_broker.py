from __future__ import annotations

import asyncio
import json
import os
import sqlite3
from pathlib import Path
from uuid import uuid4

import pytest
from aspire.sim.online_rl.artifacts import ArtifactStore
from aspire.sim.online_rl.broker.app import BrokerSettings, create_app
from aspire.sim.online_rl.broker.scheduler import ReplayScheduler
from aspire.sim.online_rl.broker.state import EpisodeService, apply_unified_patch
from aspire.sim.online_rl.broker.store import EpisodeStore
from aspire.sim.online_rl.contracts import (
    ActionRequest,
    CampaignManifest,
    CampaignTask,
    CreateEpisodeRequest,
    EpisodeBudgets,
    FinalSubmissionAction,
    FindingsRequest,
    Observation,
    ReadSearchAction,
    ReplayAction,
    TerminalScoreRequest,
    WriteEditAction,
)
from aspire.sim.online_rl.executor import (
    ExecutionArtifact,
    InitialObservationExecution,
    ReplayExecution,
)
from aspire.sim.online_rl.heldout import evaluate_frozen_code
from aspire.sim.online_rl.reward import cap_rl_episode_reward, cap_rl_seed_reward
from aspire.sim.online_rl.runner.isolation import (
    UnsafeProgramError,
    safe_builtins,
    validate_program,
)
from aspire.sim.online_rl.runner.worker import (
    _NATIVE_THREAD_ENV_VARS,
    _limit_native_threads,
    SpawnReplayExecutor,
)
from aspire.sim.online_rl.snapshots import (
    build_snapshot,
    online_rl_protocol_digest,
    sha256_file,
    verify_snapshot,
)
from aspire.sim.online_rl.task_registry import TaskRegistry
from fastapi.testclient import TestClient


class FakeExecutor:
    def observe(self, task: CampaignTask, *, timeout_seconds: int) -> InitialObservationExecution:
        del timeout_seconds
        return InitialObservationExecution(
            task_language=task.task_language,
            text="initial scene",
            artifacts=[ExecutionArtifact("initial_rgb:agentview", "image/jpeg", b"jpeg")],
        )

    def replay(
        self,
        task: CampaignTask,
        code: str,
        seed: int,
        *,
        timeout_seconds: int,
    ) -> ReplayExecution:
        del task, code, timeout_seconds
        return ReplayExecution(
            seed=seed,
            env_reward=1.0 if seed == 65 else 0.0,
            sandbox_rc=0,
            task_completed=seed == 65,
            stdout=f"seed={seed}",
            artifacts=[ExecutionArtifact("trace", "application/json", b"[]")],
        )


class FlakyHeldoutExecutor(FakeExecutor):
    def __init__(self, failures: int) -> None:
        self.failures = failures
        self.calls = 0

    def replay(
        self,
        task: CampaignTask,
        code: str,
        seed: int,
        *,
        timeout_seconds: int,
    ) -> ReplayExecution:
        self.calls += 1
        if self.calls <= self.failures:
            raise RuntimeError("transient simulator initialization failure")
        return super().replay(
            task,
            code,
            seed,
            timeout_seconds=timeout_seconds,
        )


def _registry(tmp_path: Path) -> TaskRegistry:
    env_config = tmp_path / "env.yaml"
    env_config.write_text("env: {}\n")
    skill_source = tmp_path / "skill-source"
    skill_source.mkdir()
    skill_file = skill_source / "skill.md"
    skill_file.write_text("approved skill\n")
    skill_root = tmp_path / "skills"
    skill_snapshot = build_snapshot(skill_source, skill_root)
    skill_digest = skill_snapshot["aggregate_sha256"]
    api_source = tmp_path / "api-source"
    api_source.mkdir()
    (api_source / "api.md").write_text("approved API\n")
    api_root = tmp_path / "api"
    api_snapshot = build_snapshot(api_source, api_root)
    protocol_digest = online_rl_protocol_digest(
        protocol_version="aspire-online-rl/v1",
        budgets=EpisodeBudgets().model_dump(mode="json"),
        skill_snapshot_digest=skill_digest,
        api_snapshot_digest=api_snapshot["aggregate_sha256"],
        prompt_digest="prompt-digest",
        schema_digest="tool-schema-digest",
        model_path="model",
        model_processor_digest="processor-digest",
    )
    manifest = CampaignManifest(
        campaign_id="test",
        protocol_digest=protocol_digest,
        api_snapshot=str(api_root),
        api_snapshot_digest=api_snapshot["aggregate_sha256"],
        actor_prompt_digest="prompt-digest",
        tool_schema_digest="tool-schema-digest",
        model_path="model",
        processor_digest="processor-digest",
        tasks=[
            CampaignTask(
                task_uid="libero_goal_swap/test_task",
                suite="libero_goal_swap",
                task_name="test_task",
                task_id=0,
                task_language="do the task",
                env_config=str(env_config),
                env_config_digest=sha256_file(env_config),
                skill_snapshot_id=skill_digest[:16],
                skill_snapshot=str(skill_root),
                skill_snapshot_digest=skill_digest,
                protocol_digest=protocol_digest,
            )
        ],
    )
    manifest_path = tmp_path / "campaign.json"
    manifest_path.write_text(manifest.model_dump_json())
    return TaskRegistry(manifest_path)


def _service(tmp_path: Path) -> EpisodeService:
    registry = _registry(tmp_path)
    return EpisodeService(
        registry=registry,
        store=EpisodeStore(tmp_path / "state.sqlite"),
        artifacts=ArtifactStore(tmp_path / "artifacts"),
        scheduler=ReplayScheduler(FakeExecutor(), max_concurrency=2),
        token_secret="test-secret",
        api_reference_root=registry.resolve_api_snapshot(),
    )


def test_cap_rl_examples() -> None:
    assert cap_rl_seed_reward(1.0, 0) == 1.0
    assert cap_rl_seed_reward(0.0, 0) == 0.1
    assert cap_rl_seed_reward(0.4, 0) == 0.4
    assert cap_rl_seed_reward(0.4, 1) == 0.4
    assert cap_rl_seed_reward(0.0, 1) == 0.0
    assert cap_rl_episode_reward([(0.0, 0), (1.0, 0)]) == pytest.approx(0.55)
    for invalid in (float("nan"), float("inf"), -0.1, 1.1):
        with pytest.raises(ValueError, match="finite and in"):
            cap_rl_seed_reward(invalid, 0)
    with pytest.raises(ValueError, match="at least one seed"):
        cap_rl_episode_reward([])


def test_generated_program_timeout_is_a_policy_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executor = SpawnReplayExecutor()
    task = _registry(tmp_path).get("libero_goal_swap/test_task")

    def time_out(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise TimeoutError("replay exceeded 7 seconds")

    monkeypatch.setattr(executor, "_call", time_out)
    result = executor.replay(task, "pass", 51, timeout_seconds=7)

    assert result.seed == 51
    assert result.env_reward == 0.0
    assert result.sandbox_rc == 124
    assert result.task_completed is False
    assert result.trace_summary == {
        "termination_reason": "timeout",
        "timeout_seconds": 7,
    }
    assert cap_rl_seed_reward(result.env_reward, result.sandbox_rc) == 0.0


def test_runner_retries_infrastructure_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executor = SpawnReplayExecutor()
    task = _registry(tmp_path).get("libero_goal_swap/test_task")
    expected = ReplayExecution(
        seed=51,
        env_reward=0.0,
        sandbox_rc=0,
        task_completed=False,
    )
    calls = 0

    def flaky_call(*args: object, **kwargs: object) -> ReplayExecution:
        nonlocal calls
        del args, kwargs
        calls += 1
        if calls == 1:
            raise RuntimeError("transient simulator reset failure")
        return expected

    monkeypatch.setenv("ASPIRE_RUNNER_INFRA_ATTEMPTS", "2")
    monkeypatch.setattr(executor, "_call", flaky_call)

    assert executor.replay(task, "pass", 51, timeout_seconds=7) is expected
    assert calls == 2


def test_simulator_native_thread_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ASPIRE_SIMULATOR_THREADS", "6")
    for name in _NATIVE_THREAD_ENV_VARS:
        monkeypatch.setenv(name, "224")

    _limit_native_threads()

    assert all(os.environ[name] == "6" for name in _NATIVE_THREAD_ENV_VARS)
    monkeypatch.setenv("ASPIRE_SIMULATOR_THREADS", "0")
    with pytest.raises(RuntimeError, match="between 1 and 64"):
        _limit_native_threads()


def test_generated_program_validation_blocks_raw_access() -> None:
    validate_program("import numpy as np\nprint(np.zeros(1))")
    with pytest.raises(UnsafeProgramError, match="name is not allowed"):
        validate_program("print(env.task_completed())")
    with pytest.raises(UnsafeProgramError, match="import is not allowed"):
        validate_program("import subprocess")
    with pytest.raises(UnsafeProgramError, match="private/dunder"):
        validate_program("print(str.__mro__)")
    with pytest.raises(UnsafeProgramError, match="call is not allowed"):
        validate_program("__import__('os')")


def test_safe_builtins_support_numpy_scalar_conversion() -> None:
    globals_ = {"__builtins__": safe_builtins(), "__name__": "__main__"}
    code = "import numpy as np\nresult = float(np.array([1.0, 2.0]).max())"

    exec(compile(code, "generated_robot_program.py", "exec"), globals_, globals_)

    assert globals_["result"] == 2.0


def test_strict_unified_patch() -> None:
    source = "one\ntwo\nthree\n"
    patch = "@@ -1,3 +1,3 @@\n one\n-two\n+TWO\n three\n"
    assert apply_unified_patch(source, patch) == "one\nTWO\nthree\n"


def test_episode_state_machine_reward_and_action_idempotency(tmp_path: Path) -> None:
    asyncio.run(_exercise_episode_state_machine(tmp_path))


async def _exercise_episode_state_machine(tmp_path: Path) -> None:
    service = _service(tmp_path)
    episode = await service.create_episode(
        CreateEpisodeRequest(
            task_uid="libero_goal_swap/test_task", client_episode_id=uuid4()
        )
    )
    assert episode.state_version == 0
    assert episode.initial_observation.artifacts[0].kind == "initial_rgb:agentview"

    write_request = ActionRequest(
        request_id=uuid4(),
        expected_state_version=0,
        action=WriteEditAction(
            type="write_edit", mode="replace", content="print('robot')"
        ),
    )
    written = await service.execute_action(str(episode.episode_id), write_request)
    assert written.ok
    assert written.state_version == 1
    assert await service.execute_action(str(episode.episode_id), write_request) == written
    code_sha = written.result["code_sha256"]

    replayed = await service.execute_action(
        str(episode.episode_id),
        ActionRequest(
            request_id=uuid4(),
            expected_state_version=1,
            action=ReplayAction(
                type="replay", code_sha256=code_sha, seeds=list(range(51, 66))
            ),
        ),
    )
    assert replayed.ok
    assert len(replayed.result["seeds"]) == 15

    submitted = await service.execute_action(
        str(episode.episode_id),
        ActionRequest(
            request_id=uuid4(),
            expected_state_version=2,
            action=FinalSubmissionAction(
                type="final_submission", code_sha256=code_sha
            ),
        ),
    )
    assert submitted.ok
    assert service.read_frozen_code(str(episode.episode_id)) == (
        code_sha,
        "print('robot')",
    )
    blocked_write = await service.execute_action(
        str(episode.episode_id),
        ActionRequest(
            request_id=uuid4(),
            expected_state_version=submitted.state_version,
            action=WriteEditAction(
                type="write_edit",
                mode="replace",
                content="print('changed')",
                base_sha256=code_sha,
            ),
        ),
    )
    assert blocked_write.ok is False
    assert blocked_write.error.code == "code_frozen"
    await service.save_findings(
        str(episode.episode_id), FindingsRequest(findings="deterministic findings")
    )
    score = await service.terminal_score(
        str(episode.episode_id), TerminalScoreRequest()
    )
    assert score.code_sha256 == code_sha
    assert len(score.seeds) == 15
    assert score.reward == pytest.approx((14 * 0.1 + 1.0) / 15)
    assert service.store.get_episode(str(episode.episode_id)).state.value == "completed"
    assert await service.terminal_score(
        str(episode.episode_id), TerminalScoreRequest()
    ) == score


def test_read_search_rejects_path_traversal_and_invalid_regex(tmp_path: Path) -> None:
    asyncio.run(_exercise_bad_reads(tmp_path))


def test_replay_repair_budget_is_a_policy_error(tmp_path: Path) -> None:
    asyncio.run(_exercise_replay_budget(tmp_path))


async def _exercise_replay_budget(tmp_path: Path) -> None:
    service = _service(tmp_path)
    episode = await service.create_episode(
        CreateEpisodeRequest(
            task_uid="libero_goal_swap/test_task", client_episode_id=uuid4()
        )
    )
    written = await service.execute_action(
        str(episode.episode_id),
        ActionRequest(
            request_id=uuid4(),
            expected_state_version=0,
            action=WriteEditAction(
                type="write_edit", mode="replace", content="print('robot')"
            ),
        ),
    )
    code_sha = written.result["code_sha256"]
    state_version = written.state_version
    for _ in range(4):
        replayed = await service.execute_action(
            str(episode.episode_id),
            ActionRequest(
                request_id=uuid4(),
                expected_state_version=state_version,
                action=ReplayAction(
                    type="replay", code_sha256=code_sha, seeds=[51]
                ),
            ),
        )
        assert replayed.ok
        state_version = replayed.state_version
    exhausted = await service.execute_action(
        str(episode.episode_id),
        ActionRequest(
            request_id=uuid4(),
            expected_state_version=state_version,
            action=ReplayAction(
                type="replay", code_sha256=code_sha, seeds=[51]
            ),
        ),
    )
    assert exhausted.ok is False
    assert exhausted.error.code == "replay_budget"


async def _exercise_bad_reads(tmp_path: Path) -> None:
    service = _service(tmp_path)
    episode = await service.create_episode(
        CreateEpisodeRequest(
            task_uid="libero_goal_swap/test_task", client_episode_id=uuid4()
        )
    )
    traversed = await service.execute_action(
        str(episode.episode_id),
        ActionRequest(
            request_id=uuid4(),
            expected_state_version=0,
            action=ReadSearchAction(
                type="read_search",
                root="skills",
                path="../other-episode",
            ),
        ),
    )
    assert traversed.ok is False
    assert traversed.error.code == "path_not_allowed"
    bad_regex = await service.execute_action(
        str(episode.episode_id),
        ActionRequest(
            request_id=uuid4(),
            expected_state_version=1,
            action=ReadSearchAction(
                type="read_search", root="skills", pattern="["
            ),
        ),
    )
    assert bad_regex.ok is False
    assert bad_regex.error.code == "invalid_pattern"


def test_fastapi_auth_and_artifact_contract(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    app = create_app(
        BrokerSettings(
            campaign_manifest=tmp_path / "campaign.json",
            state_db=tmp_path / "api-state.sqlite",
            artifact_root=tmp_path / "api-artifacts",
            api_reference_root=registry.resolve_api_snapshot(),
            admin_token="admin",
            token_secret="secret",
        ),
        executor=FakeExecutor(),
    )
    with TestClient(app) as client:
        assert client.get("/health/ready").json()["status"] == "ok"
        request = {
            "task_uid": "libero_goal_swap/test_task",
            "client_episode_id": str(uuid4()),
        }
        assert client.post("/v1/episodes", json=request).status_code == 401
        created_response = client.post(
            "/v1/episodes",
            json=request,
            headers={"Authorization": "Bearer admin"},
        )
        assert created_response.status_code == 200
        created = created_response.json()
        artifact = created["initial_observation"]["artifacts"][0]
        response = client.get(
            f"/v1/episodes/{created['episode_id']}/artifacts/{artifact['artifact_id']}",
            headers={"Authorization": f"Bearer {created['scoped_token']}"},
        )
        assert response.status_code == 200
        assert response.content == b"jpeg"
        second = client.post(
            "/v1/episodes",
            json={
                "task_uid": "libero_goal_swap/test_task",
                "client_episode_id": str(uuid4()),
            },
            headers={"Authorization": "Bearer admin"},
        ).json()
        cross_episode = client.get(
            f"/v1/episodes/{created['episode_id']}/artifacts/{artifact['artifact_id']}",
            headers={"Authorization": f"Bearer {second['scoped_token']}"},
        )
        assert cross_episode.status_code == 401
        action_url = f"/v1/episodes/{created['episode_id']}/actions"
        episode_headers = {
            "Authorization": f"Bearer {created['scoped_token']}"
        }
        written = client.post(
            action_url,
            json={
                "request_id": str(uuid4()),
                "expected_state_version": 0,
                "action": {
                    "type": "write_edit",
                    "mode": "replace",
                    "content": "print('frozen')",
                    "base_sha256": None,
                },
            },
            headers=episode_headers,
        ).json()
        code_sha = written["result"]["code_sha256"]
        client.post(
            action_url,
            json={
                "request_id": str(uuid4()),
                "expected_state_version": 1,
                "action": {
                    "type": "replay",
                    "code_sha256": code_sha,
                    "seeds": list(range(51, 66)),
                },
            },
            headers=episode_headers,
        ).raise_for_status()
        client.post(
            action_url,
            json={
                "request_id": str(uuid4()),
                "expected_state_version": 2,
                "action": {
                    "type": "final_submission",
                    "code_sha256": code_sha,
                },
            },
            headers=episode_headers,
        ).raise_for_status()
        exported = client.get(
            f"/v1/admin/episodes/{created['episode_id']}/frozen-code",
            headers={"Authorization": "Bearer admin"},
        )
        assert exported.status_code == 200
        assert exported.headers["x-code-sha256"] == code_sha
        assert exported.text == "print('frozen')"


def test_invalid_tool_arguments_are_audited_policy_errors(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    app = create_app(
        BrokerSettings(
            campaign_manifest=tmp_path / "campaign.json",
            state_db=tmp_path / "invalid-state.sqlite",
            artifact_root=tmp_path / "invalid-artifacts",
            api_reference_root=registry.resolve_api_snapshot(),
            admin_token="admin",
            token_secret="secret",
        ),
        executor=FakeExecutor(),
    )
    with TestClient(app) as client:
        created = client.post(
            "/v1/episodes",
            json={
                "task_uid": "libero_goal_swap/test_task",
                "client_episode_id": str(uuid4()),
            },
            headers={"Authorization": "Bearer admin"},
        ).json()
        request_id = str(uuid4())
        payload = {
            "request_id": request_id,
            "expected_state_version": 0,
            "action": {"type": "not_a_tool"},
        }
        url = f"/v1/episodes/{created['episode_id']}/actions"
        headers = {"Authorization": f"Bearer {created['scoped_token']}"}
        first = client.post(url, json=payload, headers=headers)
        second = client.post(url, json=payload, headers=headers)
        assert first.status_code == 200
        assert first.json() == second.json()
        assert first.json()["ok"] is False
        assert first.json()["error"]["code"] == "invalid_tool_arguments"
        assert first.json()["state_version"] == 1
        metrics = client.get(
            "/v1/admin/metrics",
            headers={"Authorization": "Bearer admin"},
        ).json()
        assert metrics["actions_total"] == 1
        assert metrics["invalid_tool_rate"] == 1.0


def test_snapshot_rejects_unmanifested_file(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "approved.md").write_text("approved")
    snapshot = tmp_path / "snapshot"
    build_snapshot(source, snapshot)
    snapshot.chmod(0o755)
    (snapshot / "unexpected.md").write_text("not approved")
    with pytest.raises(ValueError, match="file set differs"):
        verify_snapshot(snapshot)


def test_sqlite_restart_and_explicit_orphan_cleanup(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    database = tmp_path / "restart.sqlite"
    first = EpisodeStore(database)
    observation = Observation(task_language="do the task", text="scene")
    episode_id = str(uuid4())
    first.create_episode(
        episode_id=episode_id,
        client_episode_id=str(uuid4()),
        task_uid=registry.manifest.tasks[0].task_uid,
        observation=observation,
    )
    first.close()

    restarted = EpisodeStore(database)
    assert restarted.get_episode(episode_id).state.value == "context_ready"
    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE episodes SET updated_at='2000-01-01T00:00:00+00:00' "
            "WHERE episode_id=?",
            (episode_id,),
        )
    assert restarted.fail_orphaned_episodes(stale_after_seconds=1) == 1
    failed = restarted.get_episode(episode_id)
    assert failed.state.value == "infra_failed"
    assert restarted.fail_orphaned_episodes(stale_after_seconds=1) == 0
    assert restarted.metrics_snapshot()["events"] == {
        "infrastructure:orphan_cleanup": 1
    }
    restarted.close()


def test_heldout_evaluator_is_hash_locked_and_model_free(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    code_path = tmp_path / "fix_code.py"
    code_path.write_text("print('fixed')\n")
    manifest_path = evaluate_frozen_code(
        campaign_manifest=registry.manifest_path,
        task_uid="libero_goal_swap/test_task",
        code_path=code_path,
        output_root=tmp_path / "heldout",
        resume=False,
        seeds=(1, 2),
        executor=FakeExecutor(),
    )
    result = json.loads(manifest_path.read_text())
    assert result["status"] == "complete"
    assert result["model_calls"] == 0
    assert result["trials"] == 2
    assert result["cap_rl_mean"] == pytest.approx(0.1)
    assert result["identity"]["campaign_manifest_digest"] == sha256_file(
        registry.manifest_path
    )


def test_heldout_evaluator_retries_infrastructure_without_recording_a_trial(
    tmp_path: Path,
) -> None:
    registry = _registry(tmp_path)
    code_path = tmp_path / "fix_code.py"
    code_path.write_text("print('fixed')\n")
    executor = FlakyHeldoutExecutor(failures=2)

    manifest_path = evaluate_frozen_code(
        campaign_manifest=registry.manifest_path,
        task_uid="libero_goal_swap/test_task",
        code_path=code_path,
        output_root=tmp_path / "heldout",
        resume=False,
        seeds=(1,),
        executor=executor,
        infrastructure_retries=2,
    )

    result = json.loads(manifest_path.read_text())
    assert executor.calls == 3
    assert result["status"] == "complete"
    assert result["trials"] == 1
    assert result["model_calls"] == 0
    assert result["infrastructure_retry_limit"] == 2
    assert [event["attempt"] for event in result["infrastructure_retry_events"]] == [
        1,
        2,
    ]
