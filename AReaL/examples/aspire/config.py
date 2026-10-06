"""Configuration and cross-component safety checks for the ASPIRE example."""

from __future__ import annotations

from dataclasses import dataclass, field

from areal.api.alloc_mode import ModelAllocation
from areal.api.cli_args import GRPOConfig


def _validate_training_dispatch(batch_size: int, actor_backend: str) -> None:
    """Reject batches that the V2 DP dispatcher cannot partition.

    The inference service returns one outer trajectory per dataset row.  With
    ``export_style=individual``, all calls from the K sessions are rows inside
    that trajectory; they do not increase the outer list length seen by the
    training-service dispatcher.  Training routes deliberately do not pad, so
    the outer task-group batch must be divisible by actor DP.
    """
    dp_size = ModelAllocation.from_str(actor_backend).parallel.dp_size
    if batch_size < dp_size or batch_size % dp_size != 0:
        raise ValueError(
            "ASPIRE train_dataset.batch_size counts outer task-group trajectories "
            f"and must be >= and divisible by actor DP size; got batch_size="
            f"{batch_size}, actor DP={dp_size} ({actor_backend!r})"
        )


def _validate_long_context_rollout(
    rollout_backend: str,
    max_running_requests: int | None,
    mem_fraction_static: float | None,
) -> None:
    """Enforce the measured 131k Qwen3.5/H800 inference memory envelope."""
    tp_size = ModelAllocation.from_str(rollout_backend).parallel.tp_size
    if tp_size < 4:
        raise ValueError(
            "ASPIRE 131072-token Qwen3.5 rollout requires tensor parallelism "
            f">= 4 on 80 GiB H800; got TP={tp_size} ({rollout_backend!r})"
        )
    if max_running_requests != 4:
        raise ValueError(
            "ASPIRE requires sglang.max_running_requests=4 for the "
            "audited long-context profile"
        )
    if mem_fraction_static is None or mem_fraction_static > 0.60:
        raise ValueError(
            "ASPIRE requires sglang.mem_fraction_static <= 0.60 to reserve "
            "Qwen3.5 gated-delta-rule activation headroom"
        )


def _validate_actor_allocator(env_vars: dict[str, str]) -> None:
    """Require the allocator setting used by PyTorch, not its common typo.

    The 131k-token profile has highly variable activation sizes.  Without
    expandable segments, a worker can have ample reserved memory yet fail a
    backward allocation because those reserved blocks are fragmented.
    """
    value = env_vars.get("PYTORCH_CUDA_ALLOC_CONF", "")
    normalized = value.lower().replace(" ", "")
    if "expandable_segments:true" not in normalized:
        raise ValueError(
            "ASPIRE 131072-token FSDP training requires "
            "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True on every actor "
            "scheduling spec"
        )
    if "PYTORCH_ALLOC_CONF" in env_vars:
        raise ValueError(
            "PYTORCH_ALLOC_CONF is not a PyTorch allocator variable; use "
            "PYTORCH_CUDA_ALLOC_CONF"
        )


@dataclass
class AspireGRPOConfig(GRPOConfig):
    broker_url: str = field(default="")
    broker_admin_token_env: str = field(default="ASPIRE_BROKER_ADMIN_TOKEN")
    protocol_version: str = field(default="aspire-online-rl/v1")

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.broker_url:
            raise ValueError("broker_url is required")
        if self.protocol_version != "aspire-online-rl/v1":
            raise ValueError("this example implements only aspire-online-rl/v1")
        agent = self.rollout.agent
        if agent.mode != "inline":
            raise ValueError("ASPIRE grouped GRPO requires rollout.agent.mode='inline'")
        if agent.chat_template_type != "hf" or agent.export_style != "individual":
            raise ValueError("ASPIRE requires hf chat templates and individual export")
        if agent.turn_discount != 1.0:
            raise ValueError("ASPIRE terminal return requires turn_discount=1.0")
        if self.gconfig.n_samples != 4 or not self.gconfig.reward_normalization:
            raise ValueError("ASPIRE V1 requires K=4 and rollout reward normalization")
        if (
            self.gconfig.greedy
            or self.gconfig.temperature != 1.0
            or self.gconfig.max_new_tokens != 4096
            or self.gconfig.max_tokens != 131072
        ):
            raise ValueError(
                "ASPIRE V1 requires sampling temperature=1, max_new_tokens=4096, "
                "and max_tokens=131072"
            )
        if self.gconfig.drop_incomplete_group:
            raise ValueError("AReaL v2 requires drop_incomplete_group=false")
        if self.rollout.max_head_offpolicyness != 0:
            raise ValueError("ASPIRE V1 requires max_head_offpolicyness=0")
        if self.rollout.max_concurrent_rollouts != 4:
            raise ValueError("ASPIRE V1 requires four concurrent group members")
        if agent.engine_max_tokens != 131072 or agent.session_timeout_seconds != 18000:
            raise ValueError("ASPIRE V1 context/session limits are 131072/18000")
        if self.rollout._version != "v2" or self.actor._version != "v2":
            raise ValueError("ASPIRE requires v2 actor and rollout controllers")
        if self.actor.request_timeout < 1800:
            raise ValueError(
                "ASPIRE 131072-token PPO requires actor.request_timeout >= 1800s"
            )
        if (
            self.actor.workers_ready_timeout < 120
            or self.rollout.workers_ready_timeout < 120
        ):
            raise ValueError(
                "ASPIRE cold-start guards require actor/rollout "
                "workers_ready_timeout >= 120s"
            )
        if self.rollout.use_lora != self.actor.use_lora:
            raise ValueError("actor.use_lora and rollout.use_lora must match")
        if self.actor.use_lora and (
            self.sglang.max_loras_per_batch != self.rollout.max_concurrent_rollouts
            or self.sglang.max_loaded_loras < self.sglang.max_loras_per_batch
        ):
            raise ValueError(
                "ASPIRE LoRA requires max_loras_per_batch to equal rollout "
                "concurrency and max_loaded_loras >= max_loras_per_batch"
            )
        if self.actor.use_lora and self.actor.expected_lora_target_module_count is None:
            raise ValueError("ASPIRE LoRA requires an audited target-module count")
        if self.rollout.consumer_batch_size != self.train_dataset.batch_size:
            raise ValueError(
                "rollout.consumer_batch_size is the task-group count and must "
                "equal train_dataset.batch_size"
            )
        if self.train_dataset.batch_size < 1 or self.train_dataset.num_workers != 0:
            raise ValueError(
                "ASPIRE requires at least one task group/update and num_workers=0"
            )
        if self.actor.use_lora:
            _validate_training_dispatch(
                self.train_dataset.batch_size,
                self.actor.backend,
            )
            for scheduling_spec in self.actor.scheduling_spec:
                _validate_actor_allocator(scheduling_spec.env_vars)
            _validate_long_context_rollout(
                self.rollout.backend,
                self.sglang.max_running_requests,
                self.sglang.mem_fraction_static,
            )
        if self.total_train_epochs != 1:
            raise ValueError("ASPIRE V1 acceptance profile is exactly one epoch")
        if self.actor.loss_reduction != "episode_mean":
            raise ValueError("ASPIRE requires actor.loss_reduction='episode_mean'")
        if self.actor.kl_ctl != 0.0 or self.ref is not None or self.critic is not None:
            raise ValueError("ASPIRE V1 requires kl_ctl=0, ref=null, and critic=null")
        if self.teacher is not None:
            raise ValueError("ASPIRE V1 does not support a teacher model")
        if self.actor.reward_norm is not None or self.actor.adv_norm is not None:
            raise ValueError(
                "reward_norm and adv_norm must be null for grouped ASPIRE GRPO"
            )
        if self.actor.reward_scaling != 1.0 or self.actor.reward_bias != 0.0:
            raise ValueError("ASPIRE V1 requires reward_scaling=1 and reward_bias=0")
        if self.actor.discount != 1.0 or self.actor.gae_lambda != 1.0:
            raise ValueError("ASPIRE terminal return requires discount=gae_lambda=1")
        if self.actor.mask_no_eos_with_zero:
            raise ValueError(
                "ASPIRE trains terminal EOS; mask_no_eos_with_zero must be false"
            )
        if self.actor.recompute_logprob or self.actor.use_decoupled_loss:
            raise ValueError("ASPIRE V1 requires on-policy logprobs and coupled PPO")
        if not self.actor.fsdp.selective_loss_logits:
            raise ValueError(
                "ASPIRE 131072-token FSDP training requires "
                "actor.fsdp.selective_loss_logits=true"
            )
        if self.actor.importance_sampling_level != "token":
            raise ValueError(
                "ASPIRE selective logits requires token-level importance sampling"
            )
        if self.actor.ppo_n_minibatches != 1:
            raise ValueError("ASPIRE V1 requires one PPO minibatch")
        if self.valid_dataset is not None:
            raise ValueError(
                "held-out evaluation is external; valid_dataset must be null"
            )
