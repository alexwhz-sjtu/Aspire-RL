#!/usr/bin/env python3
"""CLI for the no-LLM held-out evaluator."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from aspire.sim.online_rl.executor import (
    RemoteReplayExecutor,
    RoundRobinReplayExecutor,
)
from aspire.sim.online_rl.heldout import evaluate_frozen_code


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-manifest", type=Path, required=True)
    parser.add_argument("--task-uid", required=True)
    parser.add_argument("--fix-code", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(1, 51)))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--runner-url", action="append")
    parser.add_argument("--runner-token-env", default="ASPIRE_RUNNER_TOKEN")
    parser.add_argument("--infra-retries", type=int, default=3)
    args = parser.parse_args()
    executor = None
    if args.runner_url:
        token = os.environ.get(args.runner_token_env)
        if not token:
            raise SystemExit(
                f"required environment variable is not set: {args.runner_token_env}"
            )
        executor = RoundRobinReplayExecutor(
            [RemoteReplayExecutor(url, token) for url in args.runner_url]
        )
    manifest = evaluate_frozen_code(
        campaign_manifest=args.campaign_manifest,
        task_uid=args.task_uid,
        code_path=args.fix_code,
        output_root=args.output_root,
        resume=args.resume,
        seeds=tuple(sorted(set(args.seeds))),
        executor=executor,
        infrastructure_retries=args.infra_retries,
    )
    print(manifest)


if __name__ == "__main__":
    main()
