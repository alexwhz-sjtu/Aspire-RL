"""Sequential multi-turn coding runtime for ASPIRE Actor rollouts."""

from __future__ import annotations

import asyncio
import base64
import copy
import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import APIStatusError, AsyncOpenAI
from openai.types.chat import ChatCompletion

from examples.aspire.tool_client import AspireToolClient, BrokerInfrastructureError

_ASSET_ROOT = Path(__file__).with_name("tool_schema.json")
TOOL_SCHEMAS: list[dict[str, Any]] = json.loads(_ASSET_ROOT.read_text())
_PROCESSOR_FILES = (
    "chat_template.jinja",
    "config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
)


def _canonical_digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def prompt_digest(system_prompt: str, findings_prompt: str) -> str:
    return _canonical_digest(
        {"findings.md": findings_prompt, "system.md": system_prompt}
    )


def tool_schema_digest() -> str:
    return _canonical_digest(TOOL_SCHEMAS)


def processor_digest(model_path: str | Path) -> str:
    root = Path(model_path).resolve()
    entries = []
    for name in _PROCESSOR_FILES:
        path = root / name
        if path.is_file():
            data = path.read_bytes()
            entries.append(
                {
                    "path": name,
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "size_bytes": len(data),
                }
            )
    if not entries:
        raise ValueError(
            f"no processor/tokenizer metadata found under model path: {root}"
        )
    return _canonical_digest(entries)


def _template_token_count(value: Any) -> int:
    if isinstance(value, Mapping):
        value = value.get("input_ids")
    if value is None:
        raise ValueError("chat template did not return input_ids")
    shape = getattr(value, "shape", None)
    if shape is not None:
        if len(shape) == 0:
            return 1
        return int(shape[-1])
    if isinstance(value, list):
        if value and isinstance(value[0], list):
            return len(value[0])
        return len(value)
    raise TypeError(f"unsupported chat-template token output: {type(value).__name__}")


@dataclass(frozen=True)
class RuntimeLimits:
    max_model_turns: int = 48
    max_output_tokens_per_turn: int = 4096
    max_policy_output_tokens: int = 65536
    context_limit: int = 131072
    findings_max_tokens: int = 2048


@dataclass(frozen=True)
class RuntimeResult:
    reward: float
    episode_id: str
    code_sha256: str | None


def _data_uri(data: bytes, media_type: str) -> str:
    encoded = base64.b64encode(data).decode()
    return f"data:{media_type};base64,{encoded}"


_CODE_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _bind_code_sha256(
    action: dict[str, Any], current_code_sha256: str | None
) -> dict[str, Any]:
    """Bind model intent to the Runtime-owned current code revision.

    Code revision identifiers are Broker state, not policy decisions. The
    model-facing tools therefore omit them. Keep this binding immediately
    before the HTTP action so the Broker remains the authority that validates
    both the injected hash and the optimistic state version.
    """
    bound = dict(action)
    action_type = bound.get("type")
    if action_type == "write_edit":
        bound["base_sha256"] = current_code_sha256
    elif action_type in {"replay", "final_submission"}:
        bound["code_sha256"] = current_code_sha256
    return bound


def _broker_code_sha256(tool_result: dict[str, Any]) -> str:
    value = (tool_result.get("result") or {}).get("code_sha256")
    if not isinstance(value, str) or _CODE_SHA256_RE.fullmatch(value) is None:
        raise BrokerInfrastructureError(
            "ASPIRE broker returned a malformed code_sha256 after write_edit"
        )
    return value


def _model_tool_result(tool_result: dict[str, Any]) -> dict[str, Any]:
    """Return an audit-preserving, context-bounded view of replay results.

    The Broker stores and returns the complete artifact manifest for every replay.
    A 15-seed replay can contain hundreds of content-addressed array and image
    entries even though the policy only needs the replay IDs and outcome summary
    before choosing ``trace_inspection``.  Feeding those manifests back into the
    chat context can consume the entire context window before a repair turn.

    Keep the authoritative Broker response untouched and remove only nested
    artifact manifests from the model-facing copy.  Artifact kind counts retain
    a cheap diagnostic hint; the replay ID remains sufficient for the approved
    trace-inspection tool to retrieve selected evidence.
    """
    result = tool_result.get("result")
    if not isinstance(result, dict):
        return tool_result

    collection_key = next(
        (
            key
            for key in ("seeds", "replays")
            if isinstance(result.get(key), list)
        ),
        None,
    )
    if collection_key is None:
        return tool_result

    compact_items = []
    for item in result[collection_key]:
        if not isinstance(item, dict):
            compact_items.append(item)
            continue
        compact_item = {key: value for key, value in item.items() if key != "artifacts"}
        artifacts = item.get("artifacts")
        if isinstance(artifacts, list):
            artifact_counts: dict[str, int] = {}
            for artifact in artifacts:
                if not isinstance(artifact, dict):
                    continue
                kind = artifact.get("kind")
                if isinstance(kind, str):
                    artifact_counts[kind] = artifact_counts.get(kind, 0) + 1
            compact_item["artifact_counts"] = artifact_counts
        compact_items.append(compact_item)

    compact_result = dict(result)
    compact_result[collection_key] = compact_items
    compact = dict(tool_result)
    compact["result"] = compact_result
    return compact


class AspireRuntime:
    def __init__(
        self,
        *,
        system_prompt: str,
        findings_prompt: str,
        model_path: str,
        limits: RuntimeLimits,
        expected_processor_digest: str,
    ) -> None:
        self.system_prompt = system_prompt
        self.findings_prompt = findings_prompt
        self.model_path = model_path
        self.limits = limits
        self.expected_prompt_digest = prompt_digest(system_prompt, findings_prompt)
        self.expected_tool_schema_digest = tool_schema_digest()
        self.expected_processor_digest = expected_processor_digest
        self._tokenizer: Any | None = None

    def prepare_context_estimator(self) -> None:
        """Load the local chat-template tokenizer outside the async rollout loop."""
        from transformers import AutoProcessor, AutoTokenizer

        try:
            processor = AutoProcessor.from_pretrained(
                self.model_path, trust_remote_code=True
            )
            self._tokenizer = getattr(processor, "tokenizer", processor)
        except (TypeError, ValueError, OSError):
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, trust_remote_code=True
            )

    def _estimate_context(self, messages: list[dict[str, Any]]) -> int | None:
        try:
            if self._tokenizer is None:
                return None
            sanitized = copy.deepcopy(messages)
            image_count = 0
            for message in sanitized:
                content = message.get("content")
                if not isinstance(content, list):
                    continue
                for part in content:
                    if part.get("type") == "image_url":
                        image_count += 1
                        part["image_url"] = {"url": "data:image/jpeg;base64,"}
            tokens = self._tokenizer.apply_chat_template(
                sanitized,
                tools=TOOL_SCHEMAS,
                tokenize=True,
                add_generation_prompt=True,
            )
            return _template_token_count(tokens) + image_count * 4096
        except Exception:
            return None

    async def _context_size(self, messages: list[dict[str, Any]]) -> int | None:
        return await asyncio.to_thread(self._estimate_context, messages)

    def _verify_episode_contract(self, episode: dict[str, Any]) -> None:
        expected = {
            "actor_prompt_digest": self.expected_prompt_digest,
            "tool_schema_digest": self.expected_tool_schema_digest,
            "processor_digest": self.expected_processor_digest,
        }
        mismatches = {
            key: {"expected": value, "broker": episode.get(key)}
            for key, value in expected.items()
            if episode.get(key) != value
        }
        if mismatches:
            raise BrokerInfrastructureError(
                "ASPIRE protocol snapshot mismatch: "
                + json.dumps(mismatches, sort_keys=True)
            )

    async def _initial_messages(
        self, episode: dict[str, Any], tools: AspireToolClient
    ) -> list[dict[str, Any]]:
        observation = episode["initial_observation"]
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    f"Task: {observation['task_language']}\n\n"
                    f"Initial observation and approved API context:\n{observation['text']}"
                ),
            }
        ]
        for reference in observation.get("artifacts", []):
            if not reference.get("media_type", "").startswith("image/"):
                continue
            artifact = await tools.artifact(reference["artifact_id"])
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": _data_uri(artifact.data, artifact.media_type),
                        "detail": "auto",
                    },
                }
            )
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": content},
        ]

    @staticmethod
    def _completion_tokens(completion: ChatCompletion) -> int:
        if (
            completion.usage is not None
            and completion.usage.completion_tokens is not None
        ):
            return int(completion.usage.completion_tokens)
        message = completion.choices[0].message
        serialized = message.model_dump_json(exclude_none=True)
        return max(1, len(serialized.encode()) // 4)

    @staticmethod
    def _context_overflow(error: APIStatusError) -> bool:
        message = str(error).lower()
        return any(
            marker in message
            for marker in (
                "context length",
                "maximum context",
                "len of prompt tokens",
                "exceeds max_total_tokens",
                "max_tokens",
                "too many tokens",
            )
        )

    async def _append_tool_images(
        self,
        messages: list[dict[str, Any]],
        tool_result: dict[str, Any],
        tools: AspireToolClient,
    ) -> None:
        result = tool_result.get("result") or {}
        references = list(result.get("artifacts") or [])
        for replay in result.get("replays") or []:
            if isinstance(replay, dict):
                references.extend(replay.get("artifacts") or [])
        image_content: list[dict[str, Any]] = []
        image_references = [
            reference
            for reference in references
            if isinstance(reference, dict)
            and reference.get("media_type", "").startswith("image/")
        ]
        for reference in image_references[:2]:
            artifact = await tools.artifact(reference["artifact_id"])
            image_content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": _data_uri(artifact.data, artifact.media_type),
                        "detail": "auto",
                    },
                }
            )
        if image_content:
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Selected diagnostic keyframes:"},
                        *image_content,
                    ],
                }
            )

    async def _auto_finalize(
        self, tools: AspireToolClient, reason: str
    ) -> RuntimeResult:
        result = await tools.terminal_score(
            auto_freeze=True,
            fallback_findings=f"Runtime auto-freeze: {reason}",
        )
        return RuntimeResult(
            reward=float(result["reward"]),
            episode_id=tools.require_episode_id(),
            code_sha256=result.get("code_sha256") or None,
        )

    async def run(
        self,
        *,
        task_uid: str,
        gateway_base_url: str,
        gateway_api_key: str,
        gateway_http_client: Any,
        tools: AspireToolClient,
    ) -> RuntimeResult:
        episode = await tools.create_episode(task_uid)
        self._verify_episode_contract(episode)
        messages = await self._initial_messages(episode, tools)
        budgets = episode.get("budgets", {})
        max_turns = min(
            self.limits.max_model_turns, int(budgets.get("max_model_turns", 48))
        )
        max_policy_tokens = min(
            self.limits.max_policy_output_tokens,
            int(budgets.get("max_policy_output_tokens", 65536)),
        )
        per_turn_tokens = min(
            self.limits.max_output_tokens_per_turn,
            int(budgets.get("max_output_tokens_per_turn", 4096)),
        )
        client = AsyncOpenAI(
            base_url=gateway_base_url,
            api_key=gateway_api_key,
            http_client=gateway_http_client,
            max_retries=0,
        )
        policy_tokens = 0
        # Episode-local authoritative pointer. AspireRuntime instances can be
        # shared across concurrent rollouts; this state must never live on self.
        current_code_sha256: str | None = None

        for turn_index in range(max_turns):
            estimate = await self._context_size(messages)
            if (
                estimate is not None
                and estimate + per_turn_tokens > self.limits.context_limit
            ):
                return await self._auto_finalize(tools, "context budget reached")
            try:
                completion: ChatCompletion = await client.chat.completions.create(
                    model="default",
                    messages=messages,
                    tools=TOOL_SCHEMAS,
                    tool_choice="auto",
                    temperature=1.0,
                    n=1,
                    stream=False,
                    store=True,
                    max_tokens=per_turn_tokens,
                )
            # The AReaL gateway normally preserves a backend 400, but backend
            # HTTPStatusError wrapping can surface the same deterministic
            # context overflow as an OpenAI 500.  Classify by the explicit
            # backend message, not by status code, so a policy-length failure
            # auto-freezes instead of discarding the whole GRPO group.  Other
            # 5xx responses remain infrastructure errors and are re-raised.
            except APIStatusError as exc:
                if self._context_overflow(exc):
                    return await self._auto_finalize(
                        tools, "gateway context limit reached"
                    )
                raise

            message = completion.choices[0].message
            messages.append(message.model_dump(mode="json", exclude_none=True))
            policy_tokens += self._completion_tokens(completion)
            calls = list(message.tool_calls or [])
            if len(calls) != 1:
                policy_error = await tools.action(
                    {
                        "type": "__invalid_tool_protocol__",
                        "observed_tool_call_count": len(calls),
                    },
                    model_turn=turn_index + 1,
                    cumulative_policy_output_tokens=policy_tokens,
                )
                if not calls:
                    messages.append(
                        {
                            "role": "user",
                            "content": json.dumps(policy_error, sort_keys=True),
                        }
                    )
                else:
                    for call in calls:
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": call.id,
                                "content": json.dumps(policy_error),
                            }
                        )
                if policy_tokens >= max_policy_tokens:
                    return await self._auto_finalize(
                        tools, "policy output token budget reached"
                    )
                continue

            call = calls[0]
            try:
                arguments = json.loads(call.function.arguments)
                if not isinstance(arguments, dict):
                    raise ValueError("tool arguments must be a JSON object")
                action = _bind_code_sha256(
                    {"type": call.function.name, **arguments},
                    current_code_sha256,
                )
                tool_result = await tools.action(
                    action,
                    model_turn=turn_index + 1,
                    cumulative_policy_output_tokens=policy_tokens,
                )
            except (json.JSONDecodeError, ValueError) as exc:
                tool_result = await tools.action(
                    {
                        "type": "__invalid_tool_arguments__",
                        "message": str(exc)[:2048],
                    },
                    model_turn=turn_index + 1,
                    cumulative_policy_output_tokens=policy_tokens,
                )
            if call.function.name == "write_edit" and tool_result.get("ok"):
                current_code_sha256 = _broker_code_sha256(tool_result)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps(
                        _model_tool_result(tool_result), sort_keys=True
                    ),
                }
            )
            await self._append_tool_images(messages, tool_result, tools)

            error = tool_result.get("error") or {}
            if error.get("code") == "episode_timeout":
                return await self._auto_finalize(
                    tools, "episode wall-time budget reached"
                )

            if call.function.name == "final_submission" and tool_result.get("ok"):
                findings_messages = [
                    *messages,
                    {"role": "user", "content": self.findings_prompt},
                ]
                findings = "Actor submitted code; reporting generation was unavailable."
                estimate = await self._context_size(findings_messages)
                if (
                    estimate is None
                    or estimate + self.limits.findings_max_tokens
                    <= self.limits.context_limit
                ):
                    try:
                        report = await client.chat.completions.create(
                            model="default",
                            messages=findings_messages,
                            temperature=1.0,
                            n=1,
                            stream=False,
                            store=False,
                            max_tokens=self.limits.findings_max_tokens,
                        )
                        findings = report.choices[0].message.content or findings
                    except Exception:
                        # Reporting is deliberately outside the trajectory and reward.
                        # A failed best-effort report must not discard a valid RL group.
                        pass
                await tools.save_findings(findings)
                score = await tools.terminal_score(auto_freeze=False)
                return RuntimeResult(
                    reward=float(score["reward"]),
                    episode_id=tools.require_episode_id(),
                    code_sha256=score.get("code_sha256") or None,
                )

            if policy_tokens >= max_policy_tokens:
                return await self._auto_finalize(
                    tools, "policy output token budget reached"
                )

        return await self._auto_finalize(tools, "model turn budget reached")
