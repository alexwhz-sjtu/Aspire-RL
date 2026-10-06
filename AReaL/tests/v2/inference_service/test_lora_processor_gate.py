"""Regression gates for versioned VLM LoRA generation through native v2 proxies."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest

from areal.api.cli_args import GenerationHyperparameters
from areal.api.io_struct import ModelRequest
from areal.v2.inference_service.data_proxy.app import (
    _create_areal_client,
    _create_inf_bridge,
)
from areal.v2.inference_service.data_proxy.config import DataProxyConfig
from areal.v2.inference_service.data_proxy.pause import PauseState
from areal.v2.inference_service.inf_bridge import InfBridge
from areal.v2.inference_service.sglang.bridge import SGLangBridgeBackend


@pytest.mark.asyncio
async def test_native_bridge_uses_loaded_lora_and_captured_version_across_resume():
    bridge = InfBridge(
        SGLangBridgeBackend(), "http://backend", PauseState(), use_lora=True, version=0
    )
    await bridge._client.aclose()
    requests = []

    def handler(request):
        payload = json.loads(request.content)
        requests.append(payload)
        first = len(requests) == 1
        # Simulate an adapter update while a paused generation is being resumed.
        if first:
            bridge.set_version(1)
        return httpx.Response(
            200,
            json={
                "meta_info": {
                    "finish_reason": {"type": "abort" if first else "stop"},
                    "output_token_logprobs": [[-0.1, 101 if first else 102]],
                }
            },
        )

    bridge._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        response = await bridge.agenerate(
            ModelRequest(
                input_ids=[1, 2],
                gconfig=GenerationHyperparameters(
                    n_samples=1, max_new_tokens=3, max_tokens=8, lora_name="actor"
                ),
            )
        )
        assert [r["lora_path"] for r in requests] == ["actor-v0", "actor-v1"]
        assert requests[1]["input_ids"] == [1, 2, 101]
        assert response.output_tokens == [101, 102]
        assert response.output_versions == [0, 1]
    finally:
        await bridge.aclose()


@pytest.mark.asyncio
async def test_text_bridge_does_not_request_lora_by_default():
    bridge = InfBridge(SGLangBridgeBackend(), "http://backend", PauseState())
    await bridge._client.aclose()
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "meta_info": {
                    "finish_reason": {"type": "stop"},
                    "output_token_logprobs": [[-0.1, 3]],
                }
            },
        )

    bridge._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        await bridge.agenerate(
            ModelRequest(
                input_ids=[1],
                gconfig=GenerationHyperparameters(
                    n_samples=1, max_new_tokens=1, max_tokens=2
                ),
            )
        )
        assert "lora_path" not in requests[0]
    finally:
        await bridge.aclose()


def test_native_proxy_constructs_processor_aware_lora_client(monkeypatch):
    monkeypatch.setenv("AREAL_V2_LORA_NAME", "actor")
    monkeypatch.setenv("AREAL_V2_REQUIRE_MULTIMODAL_PROCESSOR", "1")
    processor = object()
    tokenizer = MagicMock()
    config = DataProxyConfig(tokenizer_path="model")
    import areal.utils.hf_utils as hf_utils

    monkeypatch.setattr(
        hf_utils, "load_hf_processor_and_tokenizer", lambda _: (processor, tokenizer)
    )
    bridge = _create_inf_bridge("http://backend", PauseState(), config)
    client = _create_areal_client(bridge, SimpleNamespace(_tok=tokenizer), config)
    assert bridge.use_lora is True
    assert client.processor is processor
    assert client.chat.completions.processor is processor
    assert client.chat.completions.require_multimodal_processor is True
    assert client.chat.completions.lora_name == "actor"


def test_required_processor_loading_failure_stops_proxy(monkeypatch):
    monkeypatch.setenv("AREAL_V2_REQUIRE_MULTIMODAL_PROCESSOR", "1")
    import areal.utils.hf_utils as hf_utils

    monkeypatch.setattr(
        hf_utils, "load_hf_processor_and_tokenizer", lambda _: (None, MagicMock())
    )
    config = DataProxyConfig(tokenizer_path="broken-model")
    bridge = _create_inf_bridge("http://backend", PauseState(), config)
    with pytest.raises(ValueError, match="requires a loadable HF processor"):
        _create_areal_client(bridge, SimpleNamespace(_tok=MagicMock()), config)
