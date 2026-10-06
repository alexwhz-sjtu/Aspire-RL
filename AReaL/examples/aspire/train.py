"""Train ASPIRE Actor with AReaL v2 grouped GRPO."""

from __future__ import annotations

import sys

from examples.aspire.config import AspireGRPOConfig
from examples.aspire.dataset import load_campaign_rows

from areal import PPOTrainer
from areal.api.cli_args import load_expr_config


def main(args: list[str]) -> None:
    config, _ = load_expr_config(args, AspireGRPOConfig)
    if not config.actor.use_lora:
        raise ValueError(
            "The full-parameter profile is a deployment dry gate only; "
            "do not run a Goal-Swap campaign through examples.aspire.train."
        )
    train_dataset = load_campaign_rows(config.train_dataset.path)
    workflow_kwargs = {
        "broker_url": config.broker_url,
        "model_path": config.actor.path,
        "protocol_version": config.protocol_version,
        "broker_admin_token_env": config.broker_admin_token_env,
        "max_model_turns": 48,
        "max_output_tokens_per_turn": config.gconfig.max_new_tokens,
        "max_policy_output_tokens": 65536,
        "context_limit": config.rollout.agent.engine_max_tokens or 131072,
        "findings_max_tokens": 2048,
        "broker_request_timeout_seconds": config.rollout.agent.session_timeout_seconds,
    }
    with PPOTrainer(config, train_dataset=train_dataset, valid_dataset=None) as trainer:
        trainer.train(
            workflow="examples.aspire.agent.AspireActor",
            workflow_kwargs=workflow_kwargs,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
