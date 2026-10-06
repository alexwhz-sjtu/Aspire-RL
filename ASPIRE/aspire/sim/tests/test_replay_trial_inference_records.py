from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPLAY_TRIAL_PATH = Path(__file__).parents[1] / "scripts/libero/replay_trial.py"
SPEC = importlib.util.spec_from_file_location("replay_trial_for_tests", REPLAY_TRIAL_PATH)
assert SPEC is not None and SPEC.loader is not None
replay_trial = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = replay_trial
SPEC.loader.exec_module(replay_trial)


def _args(**overrides):
    values = {
        "suite": "libero_goal_swap",
        "task": "put_the_bowl_on_the_stove",
        "trial": 51,
        "model": "test-model",
        "server_url": "http://127.0.0.1:47000/v1/chat/completions",
        "api_key": "must-not-be-recorded",
    }
    values.update(overrides)
    return replay_trial.ReplayTrialArgs(**values)


def test_patch_libero_goal_updates_each_observation_copy():
    original_prompt = [
        {"role": "system", "content": "system"},
        {
            "role": "user",
            "content": [{"type": "text", "text": "Goal: {libero_environment_goal}"}],
        },
    ]
    obs = {"full_prompt": original_prompt}
    env = SimpleNamespace(
        low_level_env=SimpleNamespace(
            handle=SimpleNamespace(task_language="put the bowl on the stove")
        )
    )

    replay_trial._patch_libero_goal(env, obs)

    assert obs["full_prompt"][-1]["content"][0]["text"] == (
        "Goal: put the bowl on the stove"
    )
    assert original_prompt[-1]["content"][0]["text"] == (
        "Goal: {libero_environment_goal}"
    )


def test_inference_record_correlates_request_observation_response_and_code(tmp_path):
    prompt = [
        {"role": "system", "content": "generate robot code"},
        {"role": "user", "content": [{"type": "text", "text": "Goal: test"}]},
    ]
    obs = {
        "full_prompt": prompt,
        "robot_joint_pos": np.array([0.1, 0.2]),
        "agentview": {
            "pose_mat": np.eye(4),
            "intrinsics": np.eye(3),
            "images": {
                "rgb": np.zeros((4, 5, 3), dtype=np.uint8),
                "depth": np.ones((4, 5), dtype=np.float32),
            },
        },
    }
    args = _args()

    record_dir = replay_trial._start_inference_record(
        tmp_path,
        index=0,
        phase="initial",
        args=args,
        obs=obs,
        prompt=prompt,
    )
    replay_trial._finish_inference_record(
        record_dir,
        response={
            "content": "print('done')",
            "reasoning": "test reasoning",
            "elapsed_seconds": 0.25,
            "raw_response": {"id": "response-1"},
        },
        decision="initial",
        output_code="print('done')",
    )

    request = json.loads((record_dir / "request.json").read_text())
    metadata = json.loads((record_dir / "metadata.json").read_text())
    observation = json.loads((record_dir / "observation.json").read_text())
    result = json.loads((record_dir / "result.json").read_text())

    assert request["messages"] == prompt
    assert "must-not-be-recorded" not in (record_dir / "request.json").read_text()
    assert metadata["model_input_image_count"] == 0
    assert observation["cameras"]["agentview"]["rgb_shape"] == [4, 5, 3]
    assert (record_dir / "observation_agentview_rgb.png").is_file()
    assert (record_dir / "observation_agentview_depth.npy").is_file()
    assert (record_dir / "output_code.py").read_text() == "print('done')"
    assert result["decision"] == "initial"


def test_prompt_image_count_only_counts_model_request_images():
    prompt = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "inspect"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
            ],
        }
    ]

    assert replay_trial._prompt_image_count(prompt) == 1
