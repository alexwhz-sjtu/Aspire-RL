from types import SimpleNamespace

import pytest
import torch

from areal.engine.fsdp_engine import FSDPEngine, _select_token_aligned_fields


def test_select_token_aligned_fields_projects_only_token_axes() -> None:
    data = {
        "input_ids": torch.tensor([10, 11, 12, 13, 14]),
        "loss_mask": torch.tensor([0, 1, 0, 1, 0]),
        "position_ids": torch.arange(10).reshape(2, 5),
        "reward": torch.tensor([0.4]),
        "cu_seqlens": torch.tensor([0, 5], dtype=torch.int32),
        "tag": "episode",
    }

    selected = _select_token_aligned_fields(
        data, torch.tensor([1, 3]), full_length=5
    )

    assert selected["input_ids"].tolist() == [11, 13]
    assert selected["loss_mask"].tolist() == [1, 1]
    assert selected["position_ids"].tolist() == [[1, 3], [6, 8]]
    assert selected["reward"].tolist() == pytest.approx([0.4])
    assert selected["tag"] == "episode"
    assert "cu_seqlens" not in selected


def test_selective_logprobs_match_full_logits_at_selected_positions() -> None:
    torch.manual_seed(7)
    engine = object.__new__(FSDPEngine)
    engine.config = SimpleNamespace(temperature=1.0, logprobs_chunk_size=3)
    engine.parallel_helper = SimpleNamespace(
        tp_size=1,
        sp_size=1,
        tp_group=None,
    )
    input_ids = torch.tensor([[2, 5, 1, 6, 3, 4]])
    logits = torch.randn(6, 11)
    indices = torch.tensor([2, 4])

    full_logprobs, full_entropy = engine._compute_logprobs_entropy(
        logits, {"input_ids": input_ids}
    )
    selected_logprobs, selected_entropy = engine._compute_logprobs_entropy(
        logits.index_select(0, indices),
        {"input_ids": input_ids},
        selective_logits_indices=indices,
    )

    torch.testing.assert_close(selected_logprobs, full_logprobs[indices])
    torch.testing.assert_close(selected_entropy, full_entropy[indices])
