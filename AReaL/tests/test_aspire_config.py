# SPDX-License-Identifier: Apache-2.0

import pytest

from examples.aspire.config import (
    _validate_actor_allocator,
    _validate_long_context_rollout,
    _validate_training_dispatch,
)


def test_aspire_single_group_dispatch_requires_single_actor_dp_rank() -> None:
    _validate_training_dispatch(1, "fsdp:d1p1t1")

    with pytest.raises(ValueError, match=r"batch_size=1, actor DP=4"):
        _validate_training_dispatch(1, "fsdp:d4p1t1")


def test_aspire_dispatch_accepts_divisible_outer_group_batch() -> None:
    _validate_training_dispatch(4, "fsdp:d4p1t1")

    with pytest.raises(ValueError, match=r"batch_size=6, actor DP=4"):
        _validate_training_dispatch(6, "fsdp:d4p1t1")


def test_aspire_long_context_rollout_requires_tp4_and_activation_headroom() -> None:
    _validate_long_context_rollout("sglang:d1p1t4", 4, 0.60)

    with pytest.raises(ValueError, match=r"TP=2"):
        _validate_long_context_rollout("sglang:d1p1t2", 4, 0.60)
    with pytest.raises(ValueError, match=r"max_running_requests"):
        _validate_long_context_rollout("sglang:d1p1t4", 2, 0.60)
    with pytest.raises(ValueError, match=r"mem_fraction_static"):
        _validate_long_context_rollout("sglang:d1p1t4", 4, 0.70)


def test_aspire_long_context_actor_requires_cuda_expandable_segments() -> None:
    _validate_actor_allocator(
        {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    )

    with pytest.raises(ValueError, match=r"PYTORCH_CUDA_ALLOC_CONF"):
        _validate_actor_allocator(
            {"PYTORCH_ALLOC_CONF": "expandable_segments:True"}
        )
    with pytest.raises(ValueError, match=r"PYTORCH_CUDA_ALLOC_CONF"):
        _validate_actor_allocator({})
