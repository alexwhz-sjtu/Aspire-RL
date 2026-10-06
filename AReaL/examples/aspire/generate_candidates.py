"""Generate frozen ASPIRE candidates without updating model weights."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
from pathlib import Path

import httpx

from examples.aspire.agent import AspireActor
from examples.aspire.tool_client import BrokerInfrastructureError


def _write_candidates(path: Path, candidates: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(candidates, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _load_candidates(path: Path, *, resume: bool) -> dict:
    if not path.exists():
        return {}
    if not resume:
        raise FileExistsError(f"candidate output already exists: {path}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("candidate output must contain a JSON object")
    return value


def _validate_existing_candidate(task_uid: str, candidate: dict) -> None:
    code_sha256 = candidate.get("code_sha256")
    code_path = candidate.get("code_path")
    if code_sha256 is None and code_path is None:
        return
    if not isinstance(code_sha256, str) or not isinstance(code_path, str):
        raise ValueError(f"incomplete candidate metadata for {task_uid}")
    path = Path(code_path)
    if (
        not path.is_file()
        or hashlib.sha256(path.read_bytes()).hexdigest() != code_sha256
    ):
        raise ValueError(
            f"existing frozen code failed SHA256 verification for {task_uid}"
        )


async def _run(args: argparse.Namespace) -> None:
    actor = AspireActor(
        broker_url=args.broker_url,
        model_path=args.model_path,
        broker_admin_token_env=args.broker_admin_token_env,
    )
    async with httpx.AsyncClient(timeout=None) as client:
        candidates = _load_candidates(args.output, resume=args.resume)
        code_root = args.code_output_root or args.output.parent / "frozen-code"
        code_root.mkdir(parents=True, exist_ok=True)
        for task_uid in args.task_uid:
            if task_uid in candidates:
                _validate_existing_candidate(task_uid, candidates[task_uid])
                continue
            for attempt in range(args.infra_retries + 1):
                try:
                    result, code = await actor.run_with_metadata(
                        {"task_uid": task_uid},
                        base_url=args.gateway_url,
                        http_client=client,
                        api_key=args.gateway_api_key,
                        include_frozen_code=True,
                    )
                    break
                except BrokerInfrastructureError as exc:
                    if attempt >= args.infra_retries:
                        raise
                    print(
                        f"infrastructure failure for {task_uid}; retrying with a "
                        f"fresh episode ({attempt + 1}/{args.infra_retries}): {exc}",
                        file=sys.stderr,
                        flush=True,
                    )
            code_path = None
            if code is not None and result.code_sha256 is not None:
                if hashlib.sha256(code.encode()).hexdigest() != result.code_sha256:
                    raise RuntimeError(
                        "exported frozen code failed SHA256 verification"
                    )
                task_dir = code_root / task_uid.replace("/", "__")
                task_dir.mkdir(parents=True, exist_ok=True)
                target = task_dir / f"{result.episode_id}.py"
                target.write_text(code)
                code_path = str(target)
            candidates[task_uid] = {
                "episode_id": result.episode_id,
                "reward": result.reward,
                "code_sha256": result.code_sha256,
                "code_path": code_path,
            }
            _write_candidates(args.output, candidates)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--broker-url", required=True)
    parser.add_argument("--gateway-url", required=True)
    parser.add_argument(
        "--gateway-api-key", default=os.getenv("AREAL_GATEWAY_API_KEY", "")
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--broker-admin-token-env", default="ASPIRE_BROKER_ADMIN_TOKEN")
    parser.add_argument("--task-uid", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--code-output-root", type=Path)
    parser.add_argument("--infra-retries", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not args.gateway_api_key:
        raise SystemExit("gateway API key is required")
    if not 0 <= args.infra_retries <= 10:
        raise SystemExit("--infra-retries must be between 0 and 10")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
