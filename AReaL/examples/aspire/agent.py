"""AReaL v2 Agent adapter for the ASPIRE coding runtime."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from examples.aspire.runtime import (
    AspireRuntime,
    RuntimeLimits,
    RuntimeResult,
    processor_digest,
)
from examples.aspire.tool_client import AspireToolClient


class AspireActor:
    """Stateless, reentrant adapter; each ``run`` owns one broker episode."""

    def __init__(
        self,
        *,
        broker_url: str,
        model_path: str,
        protocol_version: str = "aspire-online-rl/v1",
        broker_admin_token_env: str = "ASPIRE_BROKER_ADMIN_TOKEN",
        max_model_turns: int = 48,
        max_output_tokens_per_turn: int = 4096,
        max_policy_output_tokens: int = 65536,
        context_limit: int = 131072,
        findings_max_tokens: int = 2048,
        broker_request_timeout_seconds: float = 18000.0,
    ) -> None:
        token = os.environ.get(broker_admin_token_env)
        if not token:
            raise ValueError(
                f"broker admin token environment variable is not set: {broker_admin_token_env}"
            )
        prompt_root = Path(__file__).with_name("prompts")
        self.broker_url = broker_url
        self.broker_admin_token = token
        self.protocol_version = protocol_version
        self.broker_request_timeout_seconds = broker_request_timeout_seconds
        self.runtime = AspireRuntime(
            system_prompt=(prompt_root / "system.md").read_text(),
            findings_prompt=(prompt_root / "findings.md").read_text(),
            model_path=model_path,
            limits=RuntimeLimits(
                max_model_turns=max_model_turns,
                max_output_tokens_per_turn=max_output_tokens_per_turn,
                max_policy_output_tokens=max_policy_output_tokens,
                context_limit=context_limit,
                findings_max_tokens=findings_max_tokens,
            ),
            expected_processor_digest=processor_digest(model_path),
        )
        self.runtime.prepare_context_estimator()

    async def run_with_metadata(
        self,
        data: dict[str, Any],
        *,
        base_url: str,
        http_client: Any,
        api_key: str,
        include_frozen_code: bool = False,
    ) -> tuple[RuntimeResult, str | None]:
        task_uid = data.get("task_uid")
        if not isinstance(task_uid, str) or not task_uid:
            raise ValueError("ASPIRE dataset row must contain a non-empty task_uid")
        tools = AspireToolClient(
            http_client=http_client,
            broker_url=self.broker_url,
            admin_token=self.broker_admin_token,
            protocol_version=self.protocol_version,
            request_timeout_seconds=self.broker_request_timeout_seconds,
        )
        result = await self.runtime.run(
            task_uid=task_uid,
            gateway_base_url=base_url,
            gateway_api_key=api_key,
            gateway_http_client=http_client,
            tools=tools,
        )
        code = None
        if include_frozen_code and result.code_sha256 is not None:
            exported_sha, code = await tools.frozen_code()
            if exported_sha != result.code_sha256:
                raise RuntimeError("broker exported a different frozen code hash")
        return result, code

    async def run(
        self,
        data: dict[str, Any],
        *,
        base_url: str,
        http_client: Any,
        api_key: str,
    ) -> float:
        result, _ = await self.run_with_metadata(
            data,
            base_url=base_url,
            http_client=http_client,
            api_key=api_key,
        )
        return result.reward
