# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from openai import InternalServerError
from openai.types.chat import ChatCompletion

from examples.aspire.runtime import (
    TOOL_SCHEMAS,
    AspireRuntime,
    RuntimeLimits,
    _bind_code_sha256,
    _model_tool_result,
    _template_token_count,
    prompt_digest,
    tool_schema_digest,
)

PROCESSOR_DIGEST = "processor-digest"
CODE_SHA = "a" * 64


def test_template_token_count_handles_processor_batch_encoding() -> None:
    assert _template_token_count({"input_ids": [1, 2, 3]}) == 3
    assert _template_token_count({"input_ids": [[1, 2], [3, 4]]}) == 2


def test_model_tool_schema_hides_runtime_managed_code_hashes() -> None:
    schemas = {item["function"]["name"]: item["function"] for item in TOOL_SCHEMAS}

    assert "base_sha256" not in schemas["write_edit"]["parameters"]["properties"]
    assert "code_sha256" not in schemas["replay"]["parameters"]["properties"]
    assert schemas["final_submission"]["parameters"]["properties"] == {}


def test_model_tool_result_compacts_replay_artifacts_without_mutating_audit_result() -> None:
    broker_result = {
        "ok": True,
        "state_version": 4,
        "result": {
            "seeds": [
                {
                    "replay_id": "replay-51",
                    "seed": 51,
                    "sandbox_rc": 1,
                    "task_completed": False,
                    "trace_summary": {"api_0_calls": 19},
                    "artifacts": [
                        {"artifact_id": "trace", "kind": "trace"},
                        {"artifact_id": "rgb-1", "kind": "rgb_keyframe"},
                        {"artifact_id": "rgb-2", "kind": "rgb_keyframe"},
                        {"artifact_id": "stderr", "kind": "stderr"},
                    ],
                }
            ]
        },
    }

    model_result = _model_tool_result(broker_result)

    assert "artifacts" in broker_result["result"]["seeds"][0]
    compact_seed = model_result["result"]["seeds"][0]
    assert "artifacts" not in compact_seed
    assert compact_seed["replay_id"] == "replay-51"
    assert compact_seed["trace_summary"] == {"api_0_calls": 19}
    assert compact_seed["artifact_counts"] == {
        "rgb_keyframe": 2,
        "stderr": 1,
        "trace": 1,
    }


def test_model_tool_result_leaves_non_replay_results_unchanged() -> None:
    result = {"ok": True, "result": {"code_sha256": CODE_SHA}}

    assert _model_tool_result(result) is result


def test_model_tool_result_compacts_trace_inspection_manifests() -> None:
    broker_result = {
        "ok": True,
        "result": {
            "replays": [
                {
                    "replay_id": "replay-51",
                    "artifacts": [
                        {"artifact_id": "trace", "kind": "trace"},
                        {"artifact_id": "image", "kind": "rgb_keyframe"},
                    ],
                }
            ],
            "texts": [{"kind": "trace", "text": "step summary"}],
            "truncated": False,
        },
    }

    model_result = _model_tool_result(broker_result)

    replay = model_result["result"]["replays"][0]
    assert "artifacts" not in replay
    assert replay["artifact_counts"] == {"rgb_keyframe": 1, "trace": 1}
    assert model_result["result"]["texts"] == [
        {"kind": "trace", "text": "step summary"}
    ]


def _tool_completion(index: int, name: str, arguments: str) -> ChatCompletion:
    return ChatCompletion.model_validate(
        {
            "id": f"completion-{index}",
            "created": index,
            "model": "default",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": f"call-{index}",
                                "type": "function",
                                "function": {"name": name, "arguments": arguments},
                            }
                        ],
                    },
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
    )


def _text_completion(index: int, content: str) -> ChatCompletion:
    return ChatCompletion.model_validate(
        {
            "id": f"completion-{index}",
            "created": index,
            "model": "default",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": content},
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
        }
    )


class FakeTools:
    def __init__(self) -> None:
        self.state_version = 0
        self.actions = []
        self.findings = None
        self.terminal_calls = []
        self.current_code_sha256 = None
        self.write_count = 0

    async def create_episode(self, task_uid):
        return {
            "task_uid": task_uid,
            "budgets": {
                "max_model_turns": 48,
                "max_output_tokens_per_turn": 4096,
                "max_policy_output_tokens": 65536,
            },
            "initial_observation": {
                "task_language": "put the bowl on the stove",
                "text": "scene",
                "artifacts": [],
            },
            "actor_prompt_digest": prompt_digest("system", "report"),
            "tool_schema_digest": tool_schema_digest(),
            "processor_digest": PROCESSOR_DIGEST,
        }

    async def action(self, action, **telemetry):
        del telemetry
        self.actions.append(action)
        self.state_version += 1
        if action["type"] == "write_edit":
            self.write_count += 1
            self.current_code_sha256 = chr(ord("a") + self.write_count - 1) * 64
            result = {"code_sha256": self.current_code_sha256}
        else:
            result = {"code_sha256": action["code_sha256"]}
        return {
            "ok": True,
            "state_version": self.state_version,
            "result": result,
        }

    async def artifact(self, artifact_id):
        raise AssertionError(f"unexpected artifact request: {artifact_id}")

    async def save_findings(self, findings):
        self.findings = findings

    async def terminal_score(self, *, auto_freeze, fallback_findings=None):
        self.terminal_calls.append((auto_freeze, fallback_findings))
        return {"reward": 0.4, "code_sha256": self.current_code_sha256}

    def require_episode_id(self):
        return "episode-id"


@pytest.mark.asyncio
async def test_runtime_keeps_policy_calls_and_excludes_findings(monkeypatch) -> None:
    create = AsyncMock(
        side_effect=[
            _tool_completion(
                1,
                "write_edit",
                '{"mode":"replace","content":"print(1)"}',
            ),
            _tool_completion(2, "final_submission", "{}"),
            _text_completion(3, "use a top-down grasp"),
        ]
    )
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    monkeypatch.setattr("examples.aspire.runtime.AsyncOpenAI", lambda **_: fake_client)
    runtime = AspireRuntime(
        system_prompt="system",
        findings_prompt="report",
        model_path="unused",
        limits=RuntimeLimits(),
        expected_processor_digest=PROCESSOR_DIGEST,
    )
    monkeypatch.setattr(runtime, "_estimate_context", lambda _: 1)
    tools = FakeTools()

    result = await runtime.run(
        task_uid="task",
        gateway_base_url="http://gateway",
        gateway_api_key="key",
        gateway_http_client=object(),
        tools=tools,
    )

    assert result.reward == 0.4
    assert result.episode_id == "episode-id"
    assert [action["type"] for action in tools.actions] == [
        "write_edit",
        "final_submission",
    ]
    assert tools.actions[0]["base_sha256"] is None
    assert tools.actions[1]["code_sha256"] == CODE_SHA
    assert tools.findings == "use a top-down grasp"
    assert tools.terminal_calls == [(False, None)]
    calls = create.await_args_list
    assert calls[0].kwargs["store"] is True
    assert calls[1].kwargs["store"] is True
    assert calls[2].kwargs["store"] is False
    assert "tools" not in calls[2].kwargs
    assert any(message["role"] == "tool" for message in calls[1].kwargs["messages"])


@pytest.mark.asyncio
async def test_runtime_auto_freezes_on_turn_budget(monkeypatch) -> None:
    create = AsyncMock(
        return_value=_tool_completion(
            1,
            "write_edit",
            '{"mode":"replace","content":"print(1)"}',
        )
    )
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    monkeypatch.setattr("examples.aspire.runtime.AsyncOpenAI", lambda **_: fake_client)
    runtime = AspireRuntime(
        system_prompt="system",
        findings_prompt="report",
        model_path="unused",
        limits=RuntimeLimits(max_model_turns=1),
        expected_processor_digest=PROCESSOR_DIGEST,
    )
    monkeypatch.setattr(runtime, "_estimate_context", lambda _: 1)
    tools = FakeTools()

    result = await runtime.run(
        task_uid="task",
        gateway_base_url="http://gateway",
        gateway_api_key="key",
        gateway_http_client=object(),
        tools=tools,
    )

    assert result.reward == 0.4
    assert tools.terminal_calls == [
        (True, "Runtime auto-freeze: model turn budget reached")
    ]
    assert create.await_count == 1


@pytest.mark.asyncio
async def test_runtime_auto_freezes_gateway_wrapped_context_overflow(
    monkeypatch,
) -> None:
    request = httpx.Request("POST", "http://gateway/chat/completions")
    response = httpx.Response(500, request=request)
    create = AsyncMock(
        side_effect=InternalServerError(
            "Error code: 500 - len of prompt tokens 152115 exceeds "
            "max_total_tokens 131072",
            response=response,
            body={
                "detail": "len of prompt tokens 152115 exceeds "
                "max_total_tokens 131072"
            },
        )
    )
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    monkeypatch.setattr("examples.aspire.runtime.AsyncOpenAI", lambda **_: fake_client)
    runtime = AspireRuntime(
        system_prompt="system",
        findings_prompt="report",
        model_path="unused",
        limits=RuntimeLimits(),
        expected_processor_digest=PROCESSOR_DIGEST,
    )
    monkeypatch.setattr(runtime, "_estimate_context", lambda _: 1)
    tools = FakeTools()

    result = await runtime.run(
        task_uid="task",
        gateway_base_url="http://gateway",
        gateway_api_key="key",
        gateway_http_client=object(),
        tools=tools,
    )

    assert result.reward == 0.4
    assert tools.terminal_calls == [
        (True, "Runtime auto-freeze: gateway context limit reached")
    ]
    assert create.await_count == 1


@pytest.mark.asyncio
async def test_runtime_reraises_non_context_gateway_failure(monkeypatch) -> None:
    request = httpx.Request("POST", "http://gateway/chat/completions")
    response = httpx.Response(500, request=request)
    failure = InternalServerError(
        "Error code: 500 - inference worker disconnected",
        response=response,
        body={"detail": "inference worker disconnected"},
    )
    create = AsyncMock(side_effect=failure)
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    monkeypatch.setattr("examples.aspire.runtime.AsyncOpenAI", lambda **_: fake_client)
    runtime = AspireRuntime(
        system_prompt="system",
        findings_prompt="report",
        model_path="unused",
        limits=RuntimeLimits(),
        expected_processor_digest=PROCESSOR_DIGEST,
    )
    monkeypatch.setattr(runtime, "_estimate_context", lambda _: 1)
    tools = FakeTools()

    with pytest.raises(InternalServerError, match="worker disconnected"):
        await runtime.run(
            task_uid="task",
            gateway_base_url="http://gateway",
            gateway_api_key="key",
            gateway_http_client=object(),
            tools=tools,
        )

    assert tools.terminal_calls == []


def test_runtime_binds_all_code_references_and_overwrites_model_values() -> None:
    current = "f" * 64

    assert (
        _bind_code_sha256(
            {
                "type": "write_edit",
                "mode": "replace",
                "content": "print(1)",
                "base_sha256": "model-controlled",
            },
            current,
        )["base_sha256"]
        == current
    )
    assert (
        _bind_code_sha256(
            {"type": "replay", "seeds": [51], "code_sha256": "model-controlled"},
            current,
        )["code_sha256"]
        == current
    )
    assert (
        _bind_code_sha256(
            {"type": "final_submission", "code_sha256": "model-controlled"},
            current,
        )["code_sha256"]
        == current
    )
    assert (
        _bind_code_sha256(
            {"type": "write_edit", "mode": "replace", "content": "print(1)"},
            None,
        )["base_sha256"]
        is None
    )


@pytest.mark.asyncio
async def test_runtime_uses_latest_broker_sha_for_edit_replay_and_submit(
    monkeypatch,
) -> None:
    create = AsyncMock(
        side_effect=[
            _tool_completion(1, "write_edit", '{"mode":"replace","content":"v1"}'),
            _tool_completion(2, "write_edit", '{"mode":"replace","content":"v2"}'),
            _tool_completion(3, "replay", '{"seeds":[51]}'),
            _tool_completion(4, "final_submission", "{}"),
            _text_completion(5, "findings"),
        ]
    )
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    monkeypatch.setattr("examples.aspire.runtime.AsyncOpenAI", lambda **_: fake_client)
    runtime = AspireRuntime(
        system_prompt="system",
        findings_prompt="report",
        model_path="unused",
        limits=RuntimeLimits(),
        expected_processor_digest=PROCESSOR_DIGEST,
    )
    monkeypatch.setattr(runtime, "_estimate_context", lambda _: 1)
    tools = FakeTools()

    await runtime.run(
        task_uid="task",
        gateway_base_url="http://gateway",
        gateway_api_key="key",
        gateway_http_client=object(),
        tools=tools,
    )

    assert tools.actions[0]["base_sha256"] is None
    assert tools.actions[1]["base_sha256"] == "a" * 64
    assert tools.actions[2]["code_sha256"] == "b" * 64
    assert tools.actions[3]["code_sha256"] == "b" * 64


@pytest.mark.asyncio
async def test_failed_edit_does_not_advance_runtime_managed_sha(monkeypatch) -> None:
    class RejectingEditTools(FakeTools):
        async def action(self, action, **kwargs):
            if action["type"] == "write_edit" and action["content"] == "invalid":
                self.actions.append(action)
                self.state_version += 1
                return {
                    "ok": False,
                    "state_version": self.state_version,
                    "error": {
                        "error_class": "policy",
                        "code": "invalid_code",
                        "message": "syntax validation failed",
                        "retryable": False,
                    },
                }
            return await super().action(action, **kwargs)

    create = AsyncMock(
        side_effect=[
            _tool_completion(1, "write_edit", '{"mode":"replace","content":"v1"}'),
            _tool_completion(
                2, "write_edit", '{"mode":"replace","content":"invalid"}'
            ),
            _tool_completion(3, "replay", '{"seeds":[51]}'),
            _tool_completion(4, "final_submission", "{}"),
            _text_completion(5, "findings"),
        ]
    )
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    monkeypatch.setattr("examples.aspire.runtime.AsyncOpenAI", lambda **_: fake_client)
    runtime = AspireRuntime(
        system_prompt="system",
        findings_prompt="report",
        model_path="unused",
        limits=RuntimeLimits(),
        expected_processor_digest=PROCESSOR_DIGEST,
    )
    monkeypatch.setattr(runtime, "_estimate_context", lambda _: 1)
    tools = RejectingEditTools()

    await runtime.run(
        task_uid="task",
        gateway_base_url="http://gateway",
        gateway_api_key="key",
        gateway_http_client=object(),
        tools=tools,
    )

    assert tools.actions[0]["base_sha256"] is None
    assert tools.actions[1]["base_sha256"] == "a" * 64
    assert tools.actions[2]["code_sha256"] == "a" * 64
    assert tools.actions[3]["code_sha256"] == "a" * 64
