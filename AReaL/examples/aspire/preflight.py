"""Read-only ASPIRE/AReaL protocol and model compatibility preflight."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import httpx
from openai import OpenAI

from examples.aspire.runtime import (
    TOOL_SCHEMAS,
    AspireRuntime,
    RuntimeLimits,
    processor_digest,
    prompt_digest,
    tool_schema_digest,
)


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"required environment variable is not set: {name}")
    return value


def _validate_model_fast_paths(model_config: dict, attention_backend: str) -> list[str]:
    """Import the exact symbols used by the selected Qwen training path.

    Merely finding a module spec is insufficient: ``flash-attn-4`` exposes a
    ``flash_attn`` namespace but intentionally does not provide the
    FlashAttention-2 symbols expected by Transformers.
    """

    validated: list[str] = []
    if attention_backend == "flash_attention_2":
        try:
            from flash_attn import flash_attn_func, flash_attn_varlen_func
        except (ImportError, OSError) as exc:
            raise SystemExit(
                "FlashAttention-2 is not loadable: expected flash_attn_func and "
                f"flash_attn_varlen_func ({exc})"
            ) from exc
        if not callable(flash_attn_func) or not callable(flash_attn_varlen_func):
            raise SystemExit("FlashAttention-2 entrypoints are not callable")
        validated.append("flash_attn:flash_attn_func,flash_attn_varlen_func")
    else:
        raise SystemExit(f"unsupported attention backend for this profile: {attention_backend}")

    text_config = model_config.get("text_config", model_config)
    if "linear_attention" in text_config.get("layer_types", []):
        try:
            from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
            from fla.modules import FusedRMSNormGated
            from fla.ops.gated_delta_rule import (
                chunk_gated_delta_rule,
                fused_recurrent_gated_delta_rule,
            )
        except (ImportError, OSError) as exc:
            raise SystemExit(f"Qwen linear-attention fast path is not loadable: {exc}") from exc
        symbols = (
            causal_conv1d_fn,
            causal_conv1d_update,
            FusedRMSNormGated,
            chunk_gated_delta_rule,
            fused_recurrent_gated_delta_rule,
        )
        if not all(callable(symbol) for symbol in symbols):
            raise SystemExit("Qwen linear-attention fast-path entrypoints are not callable")
        validated.extend(
            [
                "causal_conv1d:causal_conv1d_fn,causal_conv1d_update",
                "fla:FusedRMSNormGated,chunk_gated_delta_rule,"
                "fused_recurrent_gated_delta_rule",
            ]
        )
    return validated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-manifest", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--broker-url", required=True)
    parser.add_argument("--gateway-url")
    parser.add_argument("--gateway-api-key-env", default="AREAL_GATEWAY_API_KEY")
    parser.add_argument("--probe-gateway", action="store_true")
    parser.add_argument(
        "--attention-backend",
        choices=("flash_attention_2",),
        default="flash_attention_2",
    )
    args = parser.parse_args()

    root = Path(__file__).parent
    system_prompt = (root / "prompts/system.md").read_text()
    findings_prompt = (root / "prompts/findings.md").read_text()
    local_digests = {
        "actor_prompt_digest": prompt_digest(system_prompt, findings_prompt),
        "tool_schema_digest": tool_schema_digest(),
        "processor_digest": processor_digest(args.model_path),
    }
    manifest = json.loads(args.campaign_manifest.read_text())
    areal_root = Path(__file__).resolve().parents[2]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=areal_root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if manifest.get("areal_commit") != commit:
        raise SystemExit(
            f"AReaL commit mismatch: local={commit}, manifest={manifest.get('areal_commit')}"
        )
    mismatches = {
        key: {"local": value, "manifest": manifest.get(key)}
        for key, value in local_digests.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise SystemExit(f"snapshot mismatch: {json.dumps(mismatches, sort_keys=True)}")

    model_config = json.loads((args.model_path / "config.json").read_text())
    validated_fast_paths = _validate_model_fast_paths(
        model_config, args.attention_backend
    )

    runtime = AspireRuntime(
        system_prompt=system_prompt,
        findings_prompt=findings_prompt,
        model_path=str(args.model_path),
        limits=RuntimeLimits(),
        expected_processor_digest=local_digests["processor_digest"],
    )
    runtime.prepare_context_estimator()
    template_tokens = runtime._estimate_context(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Inspect the frozen API before coding."},
        ]
    )
    if template_tokens is None or template_tokens <= 0:
        raise SystemExit("local processor failed to render the tool-enabled chat template")

    ready = httpx.get(f"{args.broker_url.rstrip('/')}/health/ready", timeout=10)
    ready.raise_for_status()
    if ready.json().get("status") != "ok":
        raise SystemExit(f"broker is not ready: {ready.text}")

    result = {
        "broker_ready": True,
        "gateway_probed": False,
        "template_tokens": template_tokens,
        "attention_backend": args.attention_backend,
        "model_fast_path_symbols": validated_fast_paths,
        **local_digests,
    }
    if args.probe_gateway:
        if not args.gateway_url:
            raise SystemExit("--gateway-url is required with --probe-gateway")
        completion = OpenAI(
            base_url=args.gateway_url,
            api_key=_required_env(args.gateway_api_key_env),
            max_retries=0,
        ).chat.completions.create(
            model="default",
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": "Call read_search once for the API root.",
                },
            ],
            tools=TOOL_SCHEMAS,
            tool_choice="required",
            temperature=0.0,
            n=1,
            stream=False,
            store=False,
            # Qwen reasoning tokens count against this limit.  A 256-token
            # probe can truncate the serialized tool call after the parser has
            # emitted valid reasoning, producing a false compatibility failure.
            max_tokens=1024,
        )
        calls = completion.choices[0].message.tool_calls or []
        if len(calls) != 1 or calls[0].function.name != "read_search":
            raise SystemExit("gateway/parser probe did not produce one read_search call")
        json.loads(calls[0].function.arguments)
        result["gateway_probed"] = True

    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
