#!/usr/bin/env python3
"""Launch an authenticated ASPIRE simulator runner."""

from __future__ import annotations

import argparse
import os

import uvicorn
from aspire.sim.online_rl.runner.app import RunnerSettings, create_runner_app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8126)
    parser.add_argument("--token-env", default="ASPIRE_RUNNER_TOKEN")
    args = parser.parse_args()
    token = os.environ.get(args.token_env)
    if not token:
        raise SystemExit(f"required environment variable is not set: {args.token_env}")
    uvicorn.run(create_runner_app(RunnerSettings(token=token)), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
