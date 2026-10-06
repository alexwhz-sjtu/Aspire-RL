"""Structured, no-LLM LIBERO execution with a spawned-process boundary."""

from __future__ import annotations

import base64
import contextlib
import copy
import io
import json
import tempfile
import threading
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from aspire.sim.online_rl.contracts import CampaignTask
from aspire.sim.online_rl.runner.isolation import (
    safe_builtins,
    validate_program,
)

MAX_CONSOLE_CHARS = 1_000_000
MAX_REPLAY_ARTIFACT_BYTES = 128 * 1024 * 1024
MAX_REPLAY_ARTIFACTS = 128


class _BoundedTextBuffer(io.StringIO):
    def __init__(self, limit: int = MAX_CONSOLE_CHARS) -> None:
        super().__init__()
        self.limit = limit
        self.truncated = False

    def write(self, value: str) -> int:
        original_length = len(value)
        remaining = self.limit - self.tell()
        if remaining > 0:
            super().write(value[:remaining])
        if original_length > max(remaining, 0):
            self.truncated = True
        return original_length

    def bounded_value(self) -> str:
        value = self.getvalue()
        if self.truncated:
            value += "\n[ASPIRE console output truncated]\n"
        return value


@dataclass
class ExecutionArtifact:
    kind: str
    media_type: str
    data: bytes


@dataclass
class ReplayExecution:
    seed: int
    env_reward: float
    sandbox_rc: int
    task_completed: bool
    stdout: str = ""
    stderr: str = ""
    trace_summary: dict[str, Any] = field(default_factory=dict)
    artifacts: list[ExecutionArtifact] = field(default_factory=list)


@dataclass
class InitialObservationExecution:
    task_language: str
    text: str
    artifacts: list[ExecutionArtifact] = field(default_factory=list)


class ReplayExecutor(Protocol):
    def healthcheck(self, *, timeout_seconds: int) -> bool: ...

    def observe(
        self, task: CampaignTask, *, timeout_seconds: int
    ) -> InitialObservationExecution: ...

    def replay(
        self, task: CampaignTask, code: str, seed: int, *, timeout_seconds: int
    ) -> ReplayExecution: ...


def _find_apis(obj: Any, depth: int = 0) -> dict[str, Any]:
    if depth > 5:
        return {}
    apis = getattr(obj, "_apis", None)
    if isinstance(apis, dict):
        return apis
    for attribute in ("env", "low_level_env", "_executor", "executor"):
        child = getattr(obj, attribute, None)
        if child is not None and child is not obj:
            result = _find_apis(child, depth + 1)
            if result:
                return result
    return {}


def _build_env(task: CampaignTask) -> Any:
    import yaml

    from aspire.sim.cap.envs.configs.instantiate import instantiate

    config_path = Path(task.env_config).resolve()
    base_config = yaml.safe_load(config_path.read_text())
    config = copy.deepcopy(base_config)
    privileged = config.get("env", {}).get("cfg", {}).get("privileged", False)
    config["env"]["cfg"]["low_level"] = {
        "_target_": "aspire.sim.cap.envs.simulators.libero.FrankaLiberoEnv",
        "suite_name": task.suite,
        "task_id": task.task_id,
        "privileged": privileged,
        "max_steps": 4000,
        "seed": None,
        "enable_render": False,
        "viser_debug": False,
    }
    return instantiate(config["env"])


def observe_once(task: CampaignTask) -> InitialObservationExecution:
    from PIL import Image

    from aspire.sim.cap.utils.camera_utils import obs_get_rgb

    env = _build_env(task)
    artifacts: list[ExecutionArtifact] = []
    try:
        obs, _ = env.reset(seed=51)
        camera_views = obs_get_rgb(obs)
        text_value = json.dumps(
            {
                "task_uid": task.task_uid,
                "task_language": task.task_language,
                "observation_seed": 51,
                "camera_views": sorted(camera_views),
                "instruction": (
                    "Use read_search for the frozen skill and API reference; "
                    "the live environment prompt is intentionally not exposed."
                ),
            },
            sort_keys=True,
        )
        for camera_name, rgb in sorted(camera_views.items()):
            buffer = io.BytesIO()
            Image.fromarray(rgb).save(buffer, format="JPEG", quality=85)
            artifacts.append(
                ExecutionArtifact(
                    kind=f"initial_rgb:{camera_name}",
                    media_type="image/jpeg",
                    data=buffer.getvalue(),
                )
            )
        return InitialObservationExecution(
            task_language=task.task_language,
            text=text_value,
            artifacts=artifacts,
        )
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()


def replay_once(task: CampaignTask, code: str, seed: int) -> ReplayExecution:
    validate_program(code)
    env = _build_env(task)
    stdout_buffer = _BoundedTextBuffer()
    stderr_buffer = _BoundedTextBuffer()
    trace_summary: dict[str, Any] = {}
    artifacts: list[ExecutionArtifact] = []
    try:
        env.reset(seed=seed)
        apis = _find_apis(env)
        globals_: dict[str, Any] = {
            "__builtins__": safe_builtins(),
            "__name__": "__main__",
        }
        trace_loggers = []
        for api in apis.values():
            globals_.update(api.functions())
            get_logger = getattr(api, "get_trace_logger", None)
            if callable(get_logger):
                trace_loggers.append(get_logger())

        sandbox_rc = 0
        with (
            contextlib.redirect_stdout(stdout_buffer),
            contextlib.redirect_stderr(stderr_buffer),
        ):
            try:
                compiled = compile(code, "generated_robot_program.py", "exec")
                exec(compiled, globals_, globals_)
            except BaseException:
                sandbox_rc = 1
                traceback.print_exc(file=stderr_buffer)

        reward = float(env.compute_reward())
        low_level = getattr(env, "low_level_env", None)
        task_completed_fn = getattr(low_level, "task_completed", None)
        task_completed = bool(task_completed_fn()) if callable(task_completed_fn) else False

        with tempfile.TemporaryDirectory(prefix="aspire-online-rl-trace-") as temp_dir:
            trace_root = Path(temp_dir)
            artifact_bytes = 0
            for index, trace_logger in enumerate(trace_loggers):
                output = trace_root / f"api_{index}"
                output.mkdir(parents=True)
                trace_path = trace_logger.save(output)
                trace_summary[f"api_{index}_calls"] = len(trace_logger.entries)
                candidate_artifacts = [
                    ExecutionArtifact(
                        "trace", "application/json", trace_path.read_bytes()
                    )
                ]
                for path in sorted((output / "keyframes").glob("*")):
                    media_type = "image/jpeg" if path.suffix.lower() in {".jpg", ".jpeg"} else (
                        "image/png" if path.suffix.lower() == ".png" else "application/octet-stream"
                    )
                    if media_type.startswith("image/"):
                        kind = "rgb_keyframe"
                    elif "mask" in path.name:
                        kind = "mask"
                    elif "grasp" in path.name:
                        kind = "grasp"
                    else:
                        kind = "trace_array"
                    candidate_artifacts.append(
                        ExecutionArtifact(kind, media_type, path.read_bytes())
                    )
                for artifact in candidate_artifacts:
                    if (
                        len(artifacts) >= MAX_REPLAY_ARTIFACTS
                        or artifact_bytes + len(artifact.data)
                        > MAX_REPLAY_ARTIFACT_BYTES
                    ):
                        trace_summary["artifacts_truncated"] = True
                        continue
                    artifacts.append(artifact)
                    artifact_bytes += len(artifact.data)
                trace_logger.reset()

        return ReplayExecution(
            seed=seed,
            env_reward=reward,
            sandbox_rc=sandbox_rc,
            task_completed=task_completed,
            stdout=stdout_buffer.bounded_value(),
            stderr=stderr_buffer.bounded_value(),
            trace_summary=trace_summary,
            artifacts=artifacts,
        )
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()


class RemoteReplayExecutor:
    """Synchronous executor client used behind the broker's async scheduler."""

    def __init__(self, runner_url: str, token: str) -> None:
        if not runner_url or not token:
            raise ValueError("runner_url and token are required")
        self.runner_url = runner_url.rstrip("/")
        self.token = token

    def healthcheck(self, *, timeout_seconds: int) -> bool:
        import requests

        response = requests.get(
            f"{self.runner_url}/health/ready", timeout=timeout_seconds
        )
        response.raise_for_status()
        return response.json().get("status") == "ok"

    def _post(self, path: str, task: CampaignTask, payload: dict[str, Any], timeout: int) -> Any:
        import requests

        body = {
            "task": task.model_dump(mode="json"),
            "env_config_text": Path(task.env_config).read_text(),
            **payload,
        }
        response = requests.post(
            f"{self.runner_url}{path}",
            headers={"Authorization": f"Bearer {self.token}"},
            json=body,
            timeout=timeout + 30,
        )
        response.raise_for_status()
        return response.json()

    @staticmethod
    def _artifacts(values: list[dict[str, Any]]) -> list[ExecutionArtifact]:
        return [
            ExecutionArtifact(
                kind=value["kind"],
                media_type=value["media_type"],
                data=base64.b64decode(value["data_base64"], validate=True),
            )
            for value in values
        ]

    def observe(
        self, task: CampaignTask, *, timeout_seconds: int
    ) -> InitialObservationExecution:
        value = self._post("/internal/v1/observe", task, {}, timeout_seconds)
        return InitialObservationExecution(
            task_language=value["task_language"],
            text=value["text"],
            artifacts=self._artifacts(value.get("artifacts", [])),
        )

    def replay(
        self, task: CampaignTask, code: str, seed: int, *, timeout_seconds: int
    ) -> ReplayExecution:
        value = self._post(
            "/internal/v1/replay",
            task,
            {"code": code, "seed": seed},
            timeout_seconds,
        )
        return ReplayExecution(
            seed=int(value["seed"]),
            env_reward=float(value["env_reward"]),
            sandbox_rc=int(value["sandbox_rc"]),
            task_completed=bool(value["task_completed"]),
            stdout=value.get("stdout", ""),
            stderr=value.get("stderr", ""),
            trace_summary=value.get("trace_summary", {}),
            artifacts=self._artifacts(value.get("artifacts", [])),
        )


class RoundRobinReplayExecutor:
    """Thread-safe deterministic dispatch across independent Runner endpoints."""

    def __init__(self, executors: list[ReplayExecutor]) -> None:
        if not executors:
            raise ValueError("at least one replay executor is required")
        self._executors = tuple(executors)
        self._next_index = 0
        self._lock = threading.Lock()

    def healthcheck(self, *, timeout_seconds: int) -> bool:
        return all(
            getattr(executor, "healthcheck", lambda **_: True)(
                timeout_seconds=timeout_seconds
            )
            for executor in self._executors
        )

    def _next(self) -> ReplayExecutor:
        with self._lock:
            executor = self._executors[self._next_index]
            self._next_index = (self._next_index + 1) % len(self._executors)
            return executor

    def observe(
        self, task: CampaignTask, *, timeout_seconds: int
    ) -> InitialObservationExecution:
        return self._next().observe(task, timeout_seconds=timeout_seconds)

    def replay(
        self, task: CampaignTask, code: str, seed: int, *, timeout_seconds: int
    ) -> ReplayExecution:
        return self._next().replay(
            task, code, seed, timeout_seconds=timeout_seconds
        )
