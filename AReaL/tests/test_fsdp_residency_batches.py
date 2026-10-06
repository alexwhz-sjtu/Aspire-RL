import weakref
from types import SimpleNamespace

import torch

from areal.api.cli_args import MicroBatchSpec
from areal.engine.fsdp_engine import FSDPEngine
from areal.utils.data import MicroBatchItem, MicroBatchList


def test_two_residency_batches_transfer_together_and_release_before_next(monkeypatch):
    engine = FSDPEngine.__new__(FSDPEngine)
    engine.stream_microbatches_from_cpu = True
    engine.residency_batches = 2
    engine.device = torch.device("cpu")
    items = [{"input_ids": torch.tensor([index])} for index in range(5)]
    source = MicroBatchList(
        data={"full_batch": torch.ones(100)},
        mb_spec=MicroBatchSpec(),
        mbs=items,
        group_lens=[1] * 5,
        padded_mbs=items,
        padding_lengths=[0] * 5,
        padded_to_lengths=[1] * 5,
    )
    staged = []
    tensor_refs = []
    native = MicroBatchItem.to

    def transfer(item, device):
        # Clone creates independent storage even in this CPU-only test.
        moved = native(item, device)
        tensor = moved.padded_mb["input_ids"].clone()
        staged.append(int(tensor[0]))
        tensor_refs.append(weakref.ref(tensor))
        return moved._replace(
            orig_mb={"input_ids": tensor}, padded_mb={"input_ids": tensor}
        )

    monkeypatch.setattr(MicroBatchItem, "to", transfer)
    iterator = engine._iter_resident_microbatches(source)
    for index in range(5):
        item = next(iterator)
        assert int(item.padded_mb["input_ids"][0]) == index
        assert staged == list(range(3 if index < 3 else 5))
        if index >= 3:
            assert all(ref() is None for ref in tensor_refs[:3])
        del item
    try:
        next(iterator)
        assert False, "unexpected extra microbatch"
    except StopIteration:
        pass
    assert all(ref() is None for ref in tensor_refs)
    assert source.data["full_batch"].numel() == 100


def test_staging_replaces_original_rpc_trajectory_dictionaries(monkeypatch):
    engine = FSDPEngine.__new__(FSDPEngine)
    engine.stream_microbatches_from_cpu = True
    original = {
        "input_ids": torch.ones(2),
        "multi_modal_input": [{"pixel_values": torch.ones(3)}],
    }
    rpc_args = [original]
    nested_alias = original["multi_modal_input"]
    engine._stage_batch_input(rpc_args)
    assert rpc_args[0] is original
    assert original["multi_modal_input"] is not nested_alias
    assert original["multi_modal_input"][0]["pixel_values"].device.type == "cpu"


def test_two_residency_batches_preserve_global_gradient_and_one_optimizer_step(
    monkeypatch,
):
    from areal.engine.core import train_engine
    from areal.engine.fsdp_engine import FSDPTrainContext

    monkeypatch.setattr(train_engine.dist, "all_reduce", lambda *args, **kwargs: None)

    def run(residency):
        engine = FSDPEngine.__new__(FSDPEngine)
        engine.stream_microbatches_from_cpu = residency > 1
        engine.residency_batches = residency
        engine.device = torch.device("cpu")
        engine.dp_group = object()
        engine.parallel_helper = SimpleNamespace(dp_size=1)
        engine.enable_tree_training = False
        engine.config = SimpleNamespace(
            fsdp=SimpleNamespace(selective_loss_logits=False)
        )
        weight = torch.nn.Parameter(torch.tensor(0.5))
        optim = torch.optim.SGD([weight], lr=0.1)
        calls = []
        engine._ensure_ready = lambda: None
        engine.optimizer_zero_grad = optim.zero_grad
        engine.optimizer_step = lambda: (calls.append(1), optim.step(), {"updated": 1})[
            -1
        ]
        rows = [
            {
                "x": torch.tensor([[float(index)]]),
                "weight": torch.tensor(float(index % 2)),
            }
            for index in range(1, 6)
        ]
        mbs = MicroBatchList(
            data={},
            mb_spec=MicroBatchSpec(),
            mbs=rows,
            group_lens=[1] * 5,
            padded_mbs=rows,
            padding_lengths=[0] * 5,
            padded_to_lengths=[1] * 5,
        )
        engine._prepare_mb_list = lambda data: mbs

        def prepare(item):
            inputs = {"x": item.padded_mb["x"]}
            return inputs, FSDPTrainContext(model_inputs=inputs, mb_input=item.orig_mb)

        engine._prepare_mb_inputs = prepare
        engine.model = lambda x: SimpleNamespace(logits=x * weight)
        engine._compute_logprobs_and_loss = (
            lambda logits, ctx, loss_fn, loss_weight_fn, total_weight, **kwargs: (
                (logits - 1).square().sum() * ctx.mb_input["weight"] / total_weight
            )
        )
        result = engine.train_batch(
            {}, loss_fn=lambda: None, loss_weight_fn=lambda row: row["weight"]
        )
        assert len(calls) == 1
        assert result["num_micro_batches"] == 5
        return weight.detach()

    torch.testing.assert_close(run(2), run(1), rtol=0, atol=0)
