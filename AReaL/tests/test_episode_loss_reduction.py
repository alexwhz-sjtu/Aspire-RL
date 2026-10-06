# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from areal.api.cli_args import PPOActorConfig
from areal.experimental.openai.types import (
    InteractionWithTokenLogpReward,
    attach_episode_loss_weights,
)
from areal.trainer.ppo.actor import _all_reduce_scalar, _global_any
from areal.utils.functional import ppo_actor_loss_fn


def _cached_interaction(loss_mask: list[int]) -> InteractionWithTokenLogpReward:
    interaction = InteractionWithTokenLogpReward()
    interaction._cache = {"loss_mask": torch.tensor([loss_mask])}
    return interaction


def test_attach_episode_loss_weights_normalizes_across_calls() -> None:
    interactions = {
        "short": _cached_interaction([0, 1, 1]),
        "long": _cached_interaction([0, 1, 1, 1, 1, 1, 1, 1, 1]),
    }

    assert attach_episode_loss_weights(interactions)

    weights = [
        interaction.to_tensor_dict()["episode_loss_weights"]
        for interaction in interactions.values()
    ]
    torch.testing.assert_close(
        sum(weight.sum() for weight in weights), torch.tensor(1.0)
    )
    torch.testing.assert_close(weights[0][0, 1], torch.tensor(0.1))
    torch.testing.assert_close(weights[1][0, -1], torch.tensor(0.1))


def test_episode_mean_gives_variable_length_episodes_equal_mass() -> None:
    logprobs = torch.zeros(2, 4)
    advantages = torch.tensor([[1.0, 1.0, 0.0, 0.0], [3.0, 3.0, 3.0, 3.0]])
    loss_mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]], dtype=torch.bool)
    episode_weights = torch.tensor([[0.5, 0.5, 0.0, 0.0], [0.25, 0.25, 0.25, 0.25]])

    episode_loss, _ = ppo_actor_loss_fn(
        logprobs=logprobs,
        proximal_logprobs=logprobs,
        old_logprobs=logprobs,
        advantages=advantages,
        eps_clip=0.2,
        loss_mask=loss_mask,
        loss_weights=episode_weights,
    )
    token_loss, _ = ppo_actor_loss_fn(
        logprobs=logprobs,
        proximal_logprobs=logprobs,
        old_logprobs=logprobs,
        advantages=advantages,
        eps_clip=0.2,
        loss_mask=loss_mask,
    )

    torch.testing.assert_close(episode_loss, torch.tensor(-2.0))
    torch.testing.assert_close(token_loss, torch.tensor(-14.0 / 6.0))


def test_episode_mean_denominator_is_not_shrunk_by_effective_mask() -> None:
    values = torch.zeros(1, 2)
    loss, _ = ppo_actor_loss_fn(
        logprobs=values,
        proximal_logprobs=values,
        old_logprobs=values,
        advantages=torch.ones_like(values),
        eps_clip=0.2,
        loss_mask=torch.tensor([[1, 0]], dtype=torch.bool),
        loss_weights=torch.tensor([[0.5, 0.5]]),
    )

    torch.testing.assert_close(loss, torch.tensor(-0.5))


@pytest.mark.parametrize("value", ["sequence", "", "EPISODE_MEAN"])
def test_invalid_loss_reduction_is_rejected(value: str) -> None:
    with pytest.raises(ValueError, match="loss_reduction"):
        PPOActorConfig(loss_reduction=value)


def test_episode_mean_rejects_unsupported_surrogate() -> None:
    with pytest.raises(ValueError, match="standard PPO/GRPO"):
        PPOActorConfig(loss_reduction="episode_mean", use_sapo_loss=True)


def test_lora_target_audit_count_must_be_positive() -> None:
    with pytest.raises(ValueError, match="expected_lora_target_module_count"):
        PPOActorConfig(expected_lora_target_module_count=0)


def test_global_any_observes_nonzero_remote_rank(monkeypatch) -> None:
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(dist, "get_backend", lambda: "gloo")

    def fake_all_reduce(value, *, op) -> None:
        assert op == dist.ReduceOp.MAX
        value.fill_(1)

    monkeypatch.setattr(dist, "all_reduce", fake_all_reduce)

    assert _global_any(False)


def test_all_reduce_scalar_sums_remote_episode_weight(monkeypatch) -> None:
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(dist, "get_backend", lambda: "gloo")

    def fake_all_reduce(value, *, op) -> None:
        assert op == dist.ReduceOp.SUM
        value.add_(torch.tensor(1.0, device=value.device))

    monkeypatch.setattr(dist, "all_reduce", fake_all_reduce)

    reduced = _all_reduce_scalar(torch.tensor(0.0), op=dist.ReduceOp.SUM)
    torch.testing.assert_close(reduced.cpu(), torch.tensor(1.0))


def _run_distributed_episode_gate(rank: int, init_method: str) -> None:
    dist.init_process_group(
        backend="gloo", init_method=init_method, rank=rank, world_size=2
    )
    try:
        # Rank 0 mirrors a zero-variance task group while rank 1 mirrors a
        # trainable group. Both ranks must take the non-skip branch.
        assert _global_any(rank == 1)
        total_weight = _all_reduce_scalar(
            torch.tensor(float(rank)), op=dist.ReduceOp.SUM
        )
        torch.testing.assert_close(total_weight, torch.tensor(1.0))
    finally:
        dist.destroy_process_group()


def test_distributed_episode_gate_keeps_ranks_in_lockstep(tmp_path) -> None:
    init_file = tmp_path / "episode_gate_gloo_init"
    mp.spawn(
        _run_distributed_episode_gate,
        args=(f"file://{init_file}",),
        nprocs=2,
        join=True,
    )
