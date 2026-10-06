# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from areal.v2.inference_service.sglang.scheduler import (
    _install_qwen35_lora_module_pattern_compat,
    _qwen35_lora_hidden_dim,
    _select_lora_base_config,
)


def test_select_lora_base_config_keeps_text_only_config():
    config = SimpleNamespace(
        vocab_size=128,
        hidden_size=64,
        num_hidden_layers=2,
    )

    assert _select_lora_base_config(config) is config


def test_select_lora_base_config_unwraps_multimodal_config():
    text_config = SimpleNamespace(
        vocab_size=128,
        hidden_size=64,
        num_hidden_layers=2,
    )
    config = SimpleNamespace(text_config=text_config, vision_config=SimpleNamespace())

    assert _select_lora_base_config(config) is text_config


def test_qwen35_lora_pattern_matches_registered_and_checkpoint_names():
    from sglang.srt.models.qwen3_5 import Qwen3_5ForConditionalGeneration

    _install_qwen35_lora_module_pattern_compat()

    model = object.__new__(Qwen3_5ForConditionalGeneration)
    assert model.should_apply_lora("model.layers.3.qkv_proj")
    assert model.should_apply_lora("model.layers.3.o_proj")
    assert model.should_apply_lora("model.layers.3.self_attn.qkv_proj")
    assert model.should_apply_lora("model.layers.3.self_attn.o_proj")
    assert model.should_apply_lora("model.layers.3.mlp.gate_up_proj")
    assert model.should_apply_lora("model.layers.3.mlp.down_proj")
    assert not model.should_apply_lora("model.layers.3.input_layernorm")
    assert not model.should_apply_lora("visual.layers.3.qkv_proj")


def test_qwen35_lora_dimensions_include_gated_query_projection():
    config = SimpleNamespace(
        hidden_size=2560,
        head_dim=256,
        num_attention_heads=16,
        num_key_value_heads=4,
        attn_output_gate=True,
        intermediate_size=9728,
        vocab_size=248320,
    )

    assert _qwen35_lora_hidden_dim(config, "qkv_proj") == (2560, 10240)
    assert _qwen35_lora_hidden_dim(config, "o_proj") == (4096, 2560)
    assert _qwen35_lora_hidden_dim(config, "gate_up_proj") == (2560, 19456)
    assert _qwen35_lora_hidden_dim(config, "down_proj") == (9728, 2560)

    from sglang.srt.models.qwen3_5 import Qwen3_5ForConditionalGeneration

    _install_qwen35_lora_module_pattern_compat()
    model = object.__new__(Qwen3_5ForConditionalGeneration)
    model.config = config
    assert model.get_hidden_dim("qkv_proj", layer_idx=3) == (2560, 10240)
