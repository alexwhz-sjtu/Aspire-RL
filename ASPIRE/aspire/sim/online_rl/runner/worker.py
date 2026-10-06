"""Fresh-process execution boundary for ASPIRE simulator jobs."""

from __future__ import annotations

import multiprocessing
import os
import traceback
from typing import Any

from aspire.sim.online_rl.contracts import CampaignTask
from aspire.sim.online_rl.executor import ReplayExecution, observe_once, replay_once
from aspire.sim.online_rl.runner.isolation import scrub_sensitive_environment

_NATIVE_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def _infrastructure_attempts() -> int:
    raw_value = os.environ.get("ASPIRE_RUNNER_INFRA_ATTEMPTS", "2")
    try:
        attempts = int(raw_value)
    except ValueError as exc:
        raise RuntimeError("ASPIRE_RUNNER_INFRA_ATTEMPTS must be an integer") from exc
    if not 1 <= attempts <= 5:
        raise RuntimeError("ASPIRE_RUNNER_INFRA_ATTEMPTS must be between 1 and 5")
    return attempts


def _limit_native_threads() -> None:
    """Bound native pools before simulator dependencies are imported.

    Every replay uses a fresh spawned process, and several replays may run in
    parallel.  Leaving BLAS/OpenMP defaults at the host CPU count gives each
    child hundreds of threads and makes concurrent episodes dramatically
    slower through oversubscription.  Keep the setting process-local and
    configurable for different runner hosts.
    """

    raw_value = os.environ.get("ASPIRE_SIMULATOR_THREADS", "8")
    try:
        thread_count = int(raw_value)
    except ValueError as exc:
        raise RuntimeError("ASPIRE_SIMULATOR_THREADS must be an integer") from exc
    if not 1 <= thread_count <= 64:
        raise RuntimeError("ASPIRE_SIMULATOR_THREADS must be between 1 and 64")
    value = str(thread_count)
    for name in _NATIVE_THREAD_ENV_VARS:
        os.environ[name] = value


def _child_entry(connection: Any, mode: str, task_data: dict[str, Any], payload: Any) -> None:
    try:
        scrub_sensitive_environment()
        _limit_native_threads()
        os.environ["ASPIRE_DISABLE_MOLMO"] = "1"
        os.environ.setdefault("MUJOCO_GL", "egl")
        os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
        task = CampaignTask.model_validate(task_data)
        if mode == "observe":
            result = observe_once(task)
        else:
            code, seed = payload
            result = replay_once(task, code, seed)
        connection.send((True, result))
    except BaseException:
        connection.send((False, traceback.format_exc()))
    finally:
        connection.close()


class SpawnReplayExecutor:
    """Execute every observation/replay in a clean ``multiprocessing.spawn`` child."""

    def healthcheck(self, *, timeout_seconds: int) -> bool:
        del timeout_seconds
        return True

    def _call(self, mode: str, task: CampaignTask, payload: Any, timeout_seconds: int) -> Any:
        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe(duplex=False)
        process = context.Process(
            target=_child_entry,
            args=(child, mode, task.model_dump(mode="python"), payload),
            daemon=True,
        )
        process.start()
        child.close()
        try:
            if not parent.poll(timeout_seconds):
                process.terminate()
                process.join(timeout=10)
                if process.is_alive():
                    process.kill()
                raise TimeoutError(f"{mode} exceeded {timeout_seconds} seconds")
            ok, result = parent.recv()
            process.join(timeout=10)
            if not ok:
                raise RuntimeError(f"LIBERO {mode} failed in worker:\n{result}")
            return result
        finally:
            parent.close()
            if process.is_alive():
                process.kill()
                process.join(timeout=10)

    def observe(self, task: CampaignTask, *, timeout_seconds: int):
        last_error: RuntimeError | None = None
        for _ in range(_infrastructure_attempts()):
            try:
                return self._call("observe", task, None, timeout_seconds)
            except RuntimeError as exc:
                last_error = exc
        assert last_error is not None
        raise last_error

    def replay(self, task: CampaignTask, code: str, seed: int, *, timeout_seconds: int):
        last_error: RuntimeError | None = None
        for _ in range(_infrastructure_attempts()):
            try:
                return self._call("replay", task, (code, seed), timeout_seconds)
            except TimeoutError:
                # A generated program exhausting its per-seed wall-clock budget is a
                # policy outcome, not a runner outage.  Returning a structured
                # execution keeps the episode/group trainable while still making
                # the timeout unambiguously unsuccessful to the trusted scorer.
                return ReplayExecution(
                    seed=seed,
                    env_reward=0.0,
                    sandbox_rc=124,
                    task_completed=False,
                    stderr=f"replay exceeded {timeout_seconds} seconds",
                    trace_summary={
                        "termination_reason": "timeout",
                        "timeout_seconds": timeout_seconds,
                    },
                )
            except RuntimeError as exc:
                last_error = exc
        assert last_error is not None
        raise last_error
