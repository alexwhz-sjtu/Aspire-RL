#!/usr/bin/env python3
"""Launch the ASPIRE online-RL broker."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import uvicorn
from aspire.sim.online_rl.broker.app import BrokerSettings, create_app
from aspire.sim.online_rl.executor import (
    RemoteReplayExecutor,
    RoundRobinReplayExecutor,
)


def required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"required environment variable is not set: {name}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-manifest", type=Path, required=True)
    parser.add_argument("--state-db", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--api-reference-root", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8125)
    parser.add_argument("--max-simulator-concurrency", type=int, default=2)
    parser.add_argument(
        "--cleanup-orphans-older-than-seconds",
        type=int,
        help=(
            "On startup, mark non-terminal episodes older than this threshold "
            "as infrastructure failures. Disabled by default."
        ),
    )
    parser.add_argument("--runner-url", action="append")
    parser.add_argument("--runner-token-env", default="ASPIRE_RUNNER_TOKEN")
    args = parser.parse_args()

    executor = None
    if args.runner_url:
        token = required_env(args.runner_token_env)
        executor = RoundRobinReplayExecutor(
            [RemoteReplayExecutor(url, token) for url in args.runner_url]
        )
    settings = BrokerSettings(
        campaign_manifest=args.campaign_manifest,
        state_db=args.state_db,
        artifact_root=args.artifact_root,
        api_reference_root=args.api_reference_root,
        admin_token=required_env("ASPIRE_BROKER_ADMIN_TOKEN"),
        token_secret=required_env("ASPIRE_BROKER_TOKEN_SECRET"),
        max_simulator_concurrency=args.max_simulator_concurrency,
        cleanup_orphans_older_than_seconds=args.cleanup_orphans_older_than_seconds,
    )
    uvicorn.run(create_app(settings, executor=executor), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
