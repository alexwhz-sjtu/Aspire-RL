#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Replay a specific LIBERO trial with enhanced tracing.

Supports three modes:
  - Default (fresh LLM query): generates new code via the LLM
  - --replay-code <path>: re-executes saved code.py blocks without calling LLM
  - --interactive: drops into a Python REPL after env.reset() with all API functions

Usage:
    # Fresh LLM query
    .venv-libero/bin/python3 scripts/libero/replay_trial.py \
      --args.suite libero_goal_swap \
      --args.task "put_the_bowl_on_the_stove" \
      --args.trial 5 \
      --args.model <provider/model> \
      --args.config env_configs/libero/franka_libero_traced.yaml \
      --args.output-dir ./outputs/libero_batch_run

    # Replay saved code
    .venv-libero/bin/python3 scripts/libero/replay_trial.py \
      --args.suite libero_goal_swap \
      --args.task "put_the_bowl_on_the_stove" \
      --args.trial 5 \
      --args.replay-code ./outputs/.../trial_05_.../code.py \
      --args.config env_configs/libero/franka_libero_traced.yaml \
      --args.output-dir ./outputs/libero_batch_run

    # Interactive REPL (no LLM needed)
    .venv-libero/bin/python3 scripts/libero/replay_trial.py \
      --args.suite libero_goal \
      --args.task "put_the_bowl_on_the_stove" \
      --args.trial 3 \
      --args.interactive \
      --args.config env_configs/libero/franka_libero_traced.yaml
"""

from __future__ import annotations

import copy
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Force weights_only=False for PyTorch loading of legacy files
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
os.environ.setdefault("MUJOCO_GL", "egl")

# Ensure repo root is on PYTHONPATH
_REPO_ROOT = str(Path(__file__).resolve().parents[4])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import tyro
import yaml
from PIL import Image

from aspire.sim.cap.envs.configs.instantiate import instantiate
from aspire.sim.cap.envs.configs.loader import DictLoader


@dataclass
class ReplayTrialArgs:
    """Command-line arguments for replay_trial.py."""

    suite: str
    """LIBERO benchmark suite name (e.g. libero_goal_swap)."""

    task: str
    """LIBERO task name (e.g. put_the_bowl_on_the_stove)."""

    trial: int
    """Trial number (1-indexed). Determines env.reset(seed=trial)."""

    config: str = "env_configs/libero/franka_libero_traced.yaml"
    """Path to the YAML config."""

    output_dir: str = "./outputs/libero_batch_run"
    """Base output directory."""

    model: str = "aws/anthropic/bedrock-claude-sonnet-4-6"
    """LLM model name."""

    server_url: str = "https://inference-api.nvidia.com/v1/chat/completions"
    """LLM API endpoint."""

    api_key: str | None = None
    """API key for the LLM. Reads from NVIDIA_API_KEY env var if not provided."""

    temperature: float = 1.0
    """Sampling temperature for code generation."""

    max_tokens: int = 2048 * 10
    """Maximum tokens for LLM response."""

    reasoning_effort: str = "medium"
    """Reasoning effort for models that support it."""

    replay_code: str | None = None
    """Path to a saved code.py file. If provided, re-executes the code without LLM query."""

    interactive: bool = False
    """Drop into a Python REPL after env.reset() with all API functions in scope."""

    record_video: bool = True
    """Whether to record and save rollout video."""

    debug: bool = False
    """Enable debug logging."""

    record_inference_artifacts: bool = True
    """Record each LLM request, observation snapshot, response, and extracted code."""


def _find_task_id(suite_name: str, task_name: str) -> int:
    """Find the task ID for a given suite and task name."""
    from libero import benchmark as libero_benchmark

    benchmark_dict = libero_benchmark.get_benchmark_dict()
    if suite_name not in benchmark_dict:
        raise ValueError(f"Suite '{suite_name}' not found. Available: {list(benchmark_dict.keys())}")

    task_suite = benchmark_dict[suite_name]()
    for task_id in range(task_suite.n_tasks):
        task = task_suite.get_task(task_id)
        if task.name == task_name:
            return task_id

    # List available tasks for error message
    available = [task_suite.get_task(i).name for i in range(task_suite.n_tasks)]
    raise ValueError(f"Task '{task_name}' not found in suite '{suite_name}'. Available:\n" +
                     "\n".join(f"  - {t}" for t in available))


def _parse_code_blocks(code_path: str) -> list[str]:
    """Parse a saved code.py file into individual code blocks.

    Splits on '# Code block N' comment headers.
    """
    code_text = Path(code_path).read_text()
    # Split on "# Code block N" markers
    parts = re.split(r'^# Code block \d+\s*\n', code_text, flags=re.MULTILINE)
    # First part is usually empty (before first marker), skip it
    blocks = [p.strip() for p in parts if p.strip()]
    if not blocks:
        # No markers found — treat entire file as one block
        blocks = [code_text.strip()]
    return blocks


def _build_llm_payload(args: ReplayTrialArgs, prompt: list[dict]) -> dict[str, Any]:
    """Build the exact JSON payload sent to the OpenAI-compatible endpoint."""
    payload = {
        "model": args.model,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "messages": prompt,
    }

    # Claude models need thinking config
    from aspire.sim.cap.utils.launch_utils import CLAUDE_MODELS
    if args.model in CLAUDE_MODELS:
        payload["thinking"] = {"type": "enabled", "budget_tokens": 4096}

    return payload


def _query_llm(args: ReplayTrialArgs, prompt: list[dict]) -> dict[str, Any]:
    """Query the LLM for code generation with API key rotation."""
    import requests

    raw_key = args.api_key or os.environ.get("NVIDIA_API_KEY", "")
    if not raw_key:
        raise RuntimeError("No API key. Set NVIDIA_API_KEY or pass --api-key.")

    # Support comma-separated key pool for rotation
    api_keys = [k.strip() for k in raw_key.split(",") if k.strip()]
    key_idx = 0
    payload = _build_llm_payload(args, prompt)

    print(f"Querying LLM ({args.model})...")
    t0 = time.time()

    max_retries = len(api_keys) * 5
    for attempt in range(max_retries):
        current_key = api_keys[key_idx % len(api_keys)]
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {current_key}",
        }
        resp = requests.post(args.server_url, headers=headers, data=json.dumps(payload), timeout=200)
        if resp.status_code == 429:
            key_idx += 1
            wait = 15
            print(f"  429 rate limit on key[{key_idx-1}], rotating to key[{key_idx % len(api_keys)}], waiting {wait}s...")
            time.sleep(wait)
            continue
        elif resp.status_code in [500, 502, 503, 504]:
            wait = 30 + (attempt * 10)
            print(f"  Retry {attempt+1}: status {resp.status_code}, waiting {wait}s...")
            time.sleep(wait)
            continue
        else:
            break

    resp.raise_for_status()
    body = resp.json()
    dt = time.time() - t0
    print(f"  LLM responded in {dt:.1f}s")

    if args.debug:
        print(json.dumps(body, indent=2))

    content = body["choices"][0]["message"]["content"]
    reasoning = body["choices"][0]["message"].get("reasoning")
    return {
        "content": content,
        "reasoning": reasoning,
        "raw_response": body,
        "elapsed_seconds": dt,
    }


def _json_default(value: Any) -> Any:
    """Convert common observation values into JSON-compatible values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=_json_default) + "\n")


def _prompt_image_count(value: Any) -> int:
    """Count image content blocks that are actually present in a model request."""
    if isinstance(value, dict):
        count = int(value.get("type") in {"image", "image_url", "input_image"})
        return count + sum(_prompt_image_count(item) for item in value.values())
    if isinstance(value, list):
        return sum(_prompt_image_count(item) for item in value)
    return 0


def _new_inference_run_dir(run_dir: Path, trial: int) -> Path:
    """Create a non-overwriting directory for one trial's inference records."""
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    parent = run_dir / "inference_records"
    candidate = parent / f"trial_{trial:02d}_{timestamp}_pid{os.getpid()}"
    suffix = 1
    while candidate.exists():
        candidate = parent / f"trial_{trial:02d}_{timestamp}_pid{os.getpid()}_{suffix}"
        suffix += 1
    candidate.mkdir(parents=True)
    return candidate


def _save_rgb(path: Path, rgb: np.ndarray) -> None:
    """Save an RGB observation array without changing the model request."""
    image = np.asarray(rgb)
    if image.dtype != np.uint8:
        if np.issubdtype(image.dtype, np.floating) and image.size and image.max() <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
    Image.fromarray(image).save(path)


def _save_observation_snapshot(record_dir: Path, obs: dict[str, Any]) -> None:
    """Save full_prompt, robot state, RGB, depth, and camera calibration for one inference."""
    _write_json(record_dir / "obs_full_prompt.json", obs.get("full_prompt", []))

    snapshot: dict[str, Any] = {
        "robot_joint_pos": obs.get("robot_joint_pos"),
        "robot_cartesian_pos": obs.get("robot_cartesian_pos"),
        "cameras": {},
    }
    for camera_name in ("agentview", "robot0_eye_in_hand"):
        camera = obs.get(camera_name)
        if not isinstance(camera, dict):
            continue
        camera_record: dict[str, Any] = {
            key: camera.get(key) for key in ("pose", "pose_mat", "intrinsics")
        }
        images = camera.get("images", {})
        rgb = images.get("rgb") if isinstance(images, dict) else None
        depth = images.get("depth") if isinstance(images, dict) else None
        if isinstance(rgb, np.ndarray):
            rgb_name = f"observation_{camera_name}_rgb.png"
            _save_rgb(record_dir / rgb_name, rgb)
            camera_record["rgb_file"] = rgb_name
            camera_record["rgb_shape"] = list(rgb.shape)
        if isinstance(depth, np.ndarray):
            depth_name = f"observation_{camera_name}_depth.npy"
            np.save(record_dir / depth_name, depth)
            camera_record["depth_file"] = depth_name
            camera_record["depth_shape"] = list(depth.shape)
        snapshot["cameras"][camera_name] = camera_record
    _write_json(record_dir / "observation.json", snapshot)


def _start_inference_record(
    inference_run_dir: Path,
    *,
    index: int,
    phase: str,
    args: ReplayTrialArgs,
    obs: dict[str, Any],
    prompt: list[dict],
) -> Path:
    """Persist the request and contemporaneous observation before an LLM call."""
    record_dir = inference_run_dir / f"{index:03d}_{phase}"
    record_dir.mkdir(parents=True, exist_ok=False)
    payload = _build_llm_payload(args, prompt)
    _write_json(record_dir / "request.json", payload)
    _save_observation_snapshot(record_dir, obs)
    _write_json(
        record_dir / "metadata.json",
        {
            "index": index,
            "phase": phase,
            "model": args.model,
            "server_url": args.server_url,
            "model_input_image_count": _prompt_image_count(prompt),
            "observation_images_note": (
                "Saved for debugging. They are model inputs only when request.json "
                "contains image/image_url/input_image content blocks."
            ),
        },
    )
    return record_dir


def _finish_inference_record(
    record_dir: Path,
    *,
    response: dict[str, Any] | None = None,
    decision: str | None = None,
    output_code: str | None = None,
    error: BaseException | None = None,
) -> None:
    """Persist an LLM response and the code/decision parsed from it."""
    result: dict[str, Any] = {"decision": decision}
    if response is not None:
        result.update(
            {
                "content": response.get("content"),
                "reasoning": response.get("reasoning"),
                "elapsed_seconds": response.get("elapsed_seconds"),
            }
        )
        _write_json(record_dir / "raw_response.json", response.get("raw_response", response))
        (record_dir / "response_content.txt").write_text(str(response.get("content") or ""))
        (record_dir / "reasoning.txt").write_text(str(response.get("reasoning") or ""))
    if output_code is not None:
        (record_dir / "output_code.py").write_text(output_code)
    if error is not None:
        result["error"] = f"{type(error).__name__}: {error}"
    _write_json(record_dir / "result.json", result)


def _patch_libero_goal(env: Any, obs: dict[str, Any]) -> None:
    """Replace the LIBERO goal placeholder in every newly returned observation."""
    if "full_prompt" not in obs:
        return
    obs["full_prompt"] = copy.deepcopy(obs["full_prompt"])
    low_level_env = getattr(env, "low_level_env", None)
    handle = getattr(low_level_env, "handle", None)
    if handle is None or not hasattr(handle, "task_language"):
        return
    prompt = obs["full_prompt"]
    if not prompt:
        return
    text = prompt[-1]["content"][0]["text"]
    if "libero_environment_goal" in text:
        prompt[-1]["content"][0]["text"] = text.replace(
            "{libero_environment_goal}", handle.task_language
        )


def _extract_code(content: str) -> list[str]:
    """Extract Python code from markdown fenced blocks."""
    fence_start = "```python\n"
    fence_end = "```"
    if fence_start in content:
        start_idx = content.find(fence_start) + len(fence_start)
        content = content[start_idx:]
    if fence_end in content:
        end_idx = content.rfind(fence_end)
        content = content[:end_idx]
    return [content.strip()]


def _parse_multi_turn_decision(content: str) -> tuple[str, str | None]:
    """Parse multi-turn decision from LLM response."""
    content_stripped = content.strip()
    if content_stripped.upper().startswith("REGENERATE"):
        # Extract code after REGENERATE
        code_part = content_stripped[len("REGENERATE"):].strip()
        # Try to extract from code fences
        blocks = _extract_code(code_part)
        return "regenerate", blocks[0] if blocks else code_part
    elif content_stripped.upper().startswith("FINISH"):
        return "finish", None
    else:
        # Default: treat as regenerate with code
        blocks = _extract_code(content_stripped)
        if blocks and blocks[0]:
            return "regenerate", blocks[0]
        return "finish", None


def _find_apis(obj, depth=0):
    """Walk the env wrapper chain to find the _apis dict."""
    if depth > 5:
        return {}
    if hasattr(obj, "_apis"):
        return obj._apis
    for attr in ("env", "low_level_env", "_executor", "executor"):
        child = getattr(obj, attr, None)
        if child is not None and child is not obj:
            result = _find_apis(child, depth + 1)
            if result:
                return result
    return {}


def _run_interactive_repl(env, obs, args: ReplayTrialArgs) -> None:
    """Drop into an interactive Python REPL with all API functions in scope."""
    import code as code_module

    apis = _find_apis(env)

    # Collect all API functions into a flat namespace
    repl_ns: dict[str, Any] = {
        "env": env,
        "obs": obs,
        "np": np,
        "args": args,
    }

    fn_names = []
    trace_logger = None
    for api in apis.values():
        for fn_name, fn in api.functions().items():
            repl_ns[fn_name] = fn
            fn_names.append(fn_name)
        if hasattr(api, "get_trace_logger"):
            trace_logger = api.get_trace_logger()
            repl_ns["trace_logger"] = trace_logger

    # Helper: run a code string through env.step()
    def step(code_str: str):
        """Execute a code string through env.step() and return (obs, reward, terminated, truncated, info)."""
        return env.step(code_str)
    repl_ns["step"] = step

    # Print banner
    print("\n" + "=" * 70)
    print("  INTERACTIVE REPL — ASPIRE Trial Debugger")
    print("=" * 70)
    print(f"  Suite: {args.suite}")
    print(f"  Task:  {args.task}")
    print(f"  Trial: {args.trial} (seed={args.trial})")
    print(f"  Env reset complete. Scene is ready.")
    print()
    print("  Available API functions:")
    for fn in sorted(fn_names):
        repl_ns_fn = repl_ns[fn]
        # Get a brief description from the docstring first line
        doc = getattr(repl_ns_fn, "__doc__", "") or ""
        first_line = doc.strip().split("\n")[0][:60] if doc.strip() else ""
        print(f"    {fn}()" + (f"  — {first_line}" if first_line else ""))
    print()
    print("  Other objects in scope:")
    print("    env            — the gym environment")
    print("    obs            — last observation dict")
    print("    np             — numpy")
    if trace_logger:
        print("    trace_logger   — TraceLogger instance")
    print("    step(code_str) — execute a code string via env.step()")
    print()
    print("  Example:")
    print("    >>> obs = get_observation()")
    print("    >>> rgb = obs['agentview']['images']['rgb']")
    print("    >>> masks = segment_sam3_text_prompt(rgb, 'bowl')")
    print("    >>> print(masks[0]['score'])")
    print()
    print("  Type exit() or Ctrl-D to quit.")
    print("=" * 70 + "\n")

    code_module.interact(
        banner="",
        local=repl_ns,
        exitmsg="REPL exited.",
    )


def main(args: ReplayTrialArgs) -> None:
    """Main replay entry point."""
    mode = "Interactive REPL" if args.interactive else ("Replay code" if args.replay_code else "Fresh LLM query")
    print(f"=" * 80)
    print(f"Replay Trial {args.trial}")
    print(f"Suite: {args.suite}, Task: {args.task}")
    print(f"Mode: {mode}")
    print(f"=" * 80)

    # 1. Find task ID
    task_id = _find_task_id(args.suite, args.task)
    print(f"Found task_id={task_id} for '{args.task}' in '{args.suite}'")

    # 2. Load YAML config
    config_path = os.path.expanduser(args.config)
    with open(config_path) as f:
        base_config = yaml.safe_load(f)

    # 3. Inject task-specific low_level config
    config = copy.deepcopy(base_config)
    is_privileged = config.get("env", {}).get("cfg", {}).get("privileged", False)
    config["env"]["cfg"]["low_level"] = {
        "_target_": "aspire.sim.cap.envs.simulators.libero.FrankaLiberoEnv",
        "suite_name": args.suite,
        "task_id": task_id,
        "privileged": is_privileged,
        "max_steps": 4000,
        "seed": None,
        "enable_render": False,
        "viser_debug": False,
    }
    config["trials"] = 1

    # Build output directory: output_dir/suite/task/model_dir/run
    model_dir_name = args.model.replace("/", "_")
    run_dir = Path(args.output_dir) / args.suite / args.task / model_dir_name / "run"
    config["output_dir"] = str(run_dir)

    # Save config for reference
    config_save_dir = Path(args.output_dir) / args.suite / args.task
    config_save_dir.mkdir(parents=True, exist_ok=True)
    config_save_path = config_save_dir / "config.yaml"
    with open(config_save_path, "w") as f:
        yaml.dump(config, f)

    # 4. Instantiate environment
    print("Instantiating environment...")
    env_factory = config["env"]
    env = instantiate(env_factory)

    # 5. Reset environment with trial seed
    print(f"Resetting env with seed={args.trial}...")
    obs, info = env.reset(seed=args.trial)
    _patch_libero_goal(env, obs)

    if args.record_video and hasattr(env, "enable_video_capture"):
        env.enable_video_capture(True, clear=True)

    # 5b. Interactive REPL mode — drops into REPL and returns when done
    if args.interactive:
        _run_interactive_repl(env, obs, args)
        return

    # 6. Execute code blocks
    code_blocks: list[str] = []
    code_block_metadata: list[dict] = []
    all_responses: list[dict] = []
    num_regenerations = 0
    num_finishes = 0
    reward = 0.0
    terminated = False
    truncated = False
    info_step: dict[str, Any] = {"sandbox_rc": -1, "stdout": "", "stderr": "", "task_completed": False}
    inference_run_dir: Path | None = None
    inference_index = 0

    multi_turn_prompt_template = config["env"].get("cfg", {}).get("multi_turn_prompt")

    if args.replay_code:
        # Mode 2: Replay saved code
        print(f"Loading code from: {args.replay_code}")
        code_blocks = _parse_code_blocks(args.replay_code)
        code_block_metadata = [{"generation": 0, "regenerated": False}] * len(code_blocks)
        print(f"  Found {len(code_blocks)} code blocks")

        for i, code in enumerate(code_blocks):
            print(f"\n--- Executing code block {i} ---")
            obs_next, reward, terminated, truncated, info_step = env.step(code)
            obs = obs_next
            print(f"  reward={reward}, terminated={terminated}, task_completed={info_step.get('task_completed')}")
            if info_step.get("stderr"):
                print(f"  stderr: {info_step['stderr'][:200]}")
            if terminated or truncated:
                break

    else:
        # Mode 1: Fresh LLM query
        # The replay query is text-only; observation images are archived separately below.
        if args.record_inference_artifacts:
            inference_run_dir = _new_inference_run_dir(run_dir, args.trial)
            print(f"Inference records: {inference_run_dir}")

        initial_prompt = copy.deepcopy(obs["full_prompt"])
        initial_record = None
        if inference_run_dir is not None:
            initial_record = _start_inference_record(
                inference_run_dir,
                index=inference_index,
                phase="initial",
                args=args,
                obs=obs,
                prompt=initial_prompt,
            )
        print("Querying LLM for initial code...")
        try:
            llm_response = _query_llm(args, initial_prompt)
        except BaseException as error:
            if initial_record is not None:
                _finish_inference_record(initial_record, error=error)
            raise
        raw_code = llm_response["content"]

        initial_blocks = _extract_code(raw_code)
        initial_code = "\n\n".join(initial_blocks)
        if initial_record is not None:
            _finish_inference_record(
                initial_record,
                response=llm_response,
                decision="initial",
                output_code=initial_code,
            )
            inference_index += 1
        code_blocks.extend(initial_blocks)
        code_block_metadata.extend([{"generation": 0, "regenerated": False}] * len(initial_blocks))
        all_responses.append({
            "block_idx": [0],
            "code_blocks": initial_blocks,
            "decision": "initial",
            "initial_prompt": initial_prompt,
            "reasoning": llm_response.get("reasoning", ""),
            "inference_record": (
                str(initial_record.relative_to(run_dir)) if initial_record is not None else None
            ),
        })

        code_block_idx = 0
        max_turns = 30

        while code_block_idx < len(code_blocks) and code_block_idx <= max_turns:
            code = code_blocks[code_block_idx]
            code_block_idx += 1

            print(f"\n--- Executing code block {code_block_idx - 1} ---")
            obs_next, reward, terminated, truncated, info_step = env.step(code)
            obs = obs_next
            _patch_libero_goal(env, obs)

            print(f"  reward={reward}, terminated={terminated}, task_completed={info_step.get('task_completed')}")
            if info_step.get("stderr"):
                print(f"  stderr: {info_step['stderr'][:200]}")

            # Multi-turn decision
            if multi_turn_prompt_template:
                if "terminated episode" in info_step.get("stderr", ""):
                    truncated = True
                    break

                executed_code = "\n".join(code_blocks[:code_block_idx])
                complete_mt_prompt = multi_turn_prompt_template.format(
                    executed_code=executed_code,
                    console_stdout=info_step["stdout"],
                    console_stderr=info_step["stderr"],
                )

                mt_prompt = copy.deepcopy(obs["full_prompt"])
                mt_prompt.append({
                    "role": "user",
                    "content": [{"type": "text", "text": complete_mt_prompt}],
                })

                mt_record = None
                if inference_run_dir is not None:
                    mt_record = _start_inference_record(
                        inference_run_dir,
                        index=inference_index,
                        phase=f"multiturn_after_block_{code_block_idx - 1:02d}",
                        args=args,
                        obs=obs,
                        prompt=mt_prompt,
                    )
                try:
                    mt_response = _query_llm(args, mt_prompt)
                except BaseException as error:
                    if mt_record is not None:
                        _finish_inference_record(mt_record, error=error)
                    raise
                decision, new_code = _parse_multi_turn_decision(mt_response["content"])
                if mt_record is not None:
                    _finish_inference_record(
                        mt_record,
                        response=mt_response,
                        decision=decision,
                        output_code=new_code,
                    )
                    inference_index += 1

                response_record = {
                    "block_idx": [code_block_idx],
                    "code_blocks": [new_code] if new_code else [],
                    "decision": decision,
                    "multi_turn_prompt": mt_prompt,
                    "reasoning": mt_response.get("reasoning", ""),
                    "inference_record": (
                        str(mt_record.relative_to(run_dir)) if mt_record is not None else None
                    ),
                }

                if decision == "regenerate" and new_code:
                    print("  LLM chose: REGENERATE")
                    new_blocks = _extract_code(new_code) if "```" in new_code else [new_code]
                    response_record["code_blocks"] = new_blocks
                    all_responses.append(response_record)
                    del code_blocks[code_block_idx:]
                    del code_block_metadata[code_block_idx:]
                    code_blocks.extend(new_blocks)
                    code_block_metadata.extend([{
                        "generation": num_regenerations + 1,
                        "regenerated": True,
                    }] * len(new_blocks))
                    num_regenerations += 1
                elif decision == "finish":
                    print("  LLM chose: FINISH")
                    all_responses.append(response_record)
                    num_finishes += 1
                    break
                else:
                    all_responses.append(response_record)

    # 7. Build final annotated code
    annotated_blocks = []
    for i, (block, meta) in enumerate(zip(code_blocks, code_block_metadata)):
        annotated_blocks.append(f"# Code block {i}\n{block}")
    final_code = "\n\n".join(annotated_blocks)

    # 8. Build log/summary
    sandbox_rc = info_step.get("sandbox_rc", 0)
    task_completed = info_step.get("task_completed", False)

    log_lines = [
        "-" * 100,
        f"Trial {args.trial} — {args.suite}/{args.task}",
        f"Mode: {'replay' if args.replay_code else 'fresh LLM query'}",
        f"Model: {args.model}",
        "Generated program:",
        final_code,
        "",
        "Environment response:",
        f"  Sandbox failed: {sandbox_rc}",
        f"  Stdout: {info_step.get('stdout', '')}",
        f"  Stderr: {info_step.get('stderr', '')}",
        f"  Reward: {reward}",
        f"  Task Completed: {task_completed}",
        f"  Terminated: {terminated}, Truncated: {truncated}",
        f"  Num Regenerations: {num_regenerations}",
        f"  Num Finishes: {num_finishes}",
        f"  Num Code Blocks: {len(code_blocks)}",
        f"  Inference Records: {inference_run_dir or 'disabled/not applicable'}",
        "-" * 100,
    ]

    # 9. Save artifacts
    trial_dir = run_dir / f"trial_{args.trial:02d}_sandboxrc_{sandbox_rc}_reward_{reward:.3f}_taskcompleted_{int(task_completed or False)}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    (trial_dir / "code.py").write_text(final_code)
    (trial_dir / "summary.txt").write_text("\n".join(log_lines))
    if all_responses:
        (trial_dir / "all_responses.json").write_text(json.dumps(all_responses, indent=2))
    if inference_run_dir is not None:
        (trial_dir / "inference_records_path.txt").write_text(str(inference_run_dir) + "\n")

    # Save trace from the traced API
    apis = _find_apis(env)
    if apis:
        for api in apis.values():
            if hasattr(api, "get_trace_logger"):
                trace_logger = api.get_trace_logger()
                trace_logger.save(str(trial_dir))
                trace_logger.reset()
                print(f"  Saved enhanced trace (depth/mask/grasp arrays) to {trial_dir}/keyframes/")
    else:
        print("  Warning: could not find _apis for trace saving")

    # Save video + evenly-spaced video keyframes
    if args.record_video and hasattr(env, "get_video_frames"):
        frames = env.get_video_frames(clear=True)
        if frames:
            import imageio.v2 as imageio
            video_path = trial_dir / f"video_{reward:.3f}.mp4"
            with imageio.get_writer(str(video_path), fps=30, format="FFMPEG", codec="libx264") as writer:
                for frame in frames:
                    writer.append_data(np.ascontiguousarray(frame))
            print(f"Saved video: {video_path} ({len(frames)} frames)")

            # Save 10 evenly-spaced frames so keyframes cover the full rollout
            from PIL import Image as _Image
            n_keyframes = 10
            keyframes_dir = trial_dir / "keyframes"
            keyframes_dir.mkdir(parents=True, exist_ok=True)
            n_kf = min(n_keyframes, len(frames))
            indices = [int(i * (len(frames) - 1) / max(n_kf - 1, 1)) for i in range(n_kf)]
            for rank, idx in enumerate(indices):
                img = _Image.fromarray(frames[idx])
                img.save(keyframes_dir / f"video_frame_{rank:02d}_of_{n_keyframes}_step_{idx:04d}.jpg", quality=85)
            print(f"Saved {n_keyframes} video keyframes to {keyframes_dir}")

    print(f"\n{'=' * 80}")
    print(f"Trial {args.trial} complete!")
    print(f"  Reward: {reward}")
    print(f"  Task Completed: {task_completed}")
    print(f"  Output: {trial_dir}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    tyro.cli(main)
