# SPDX-License-Identifier: Apache-2.0

import sys

from areal.api.cli_args import SGLangConfig, get_py_cmd


def test_get_py_cmd_uses_active_interpreter():
    cmd = get_py_cmd("example.module", {"port": 1234, "enabled": True})

    assert cmd == [
        sys.executable,
        "-m",
        "example.module",
        "--port",
        "1234",
        "--enabled",
    ]


def test_sglang_command_exposes_lora_batch_limit():
    cmd = SGLangConfig.build_cmd_from_args(
        {"model_path": "model", "max_loras_per_batch": 4}
    )

    assert cmd[0] == sys.executable
    assert cmd[-2:] == ["--max-loras-per-batch", "4"]
