import re

import pytest

torch = pytest.importorskip("torch")

from config import RunConfig
from model_utils import (
    build_lora_config,
    count_trainable_parameters,
    generation_kwargs,
    resolve_torch_device,
    resolve_torch_dtype,
)


def test_lora_config_matches_the_spec():
    lora = build_lora_config(RunConfig())
    assert (lora.r, lora.lora_alpha, lora.lora_dropout) == (32, 32, 0.05)
    assert lora.task_type == "CAUSAL_LM"
    assert lora.bias == "none"


def test_lora_target_regex_covers_text_decoder_projections_but_not_vision():
    pattern = re.compile(RunConfig().lora_target_modules)
    must_match = [
        "model.language_model.layers.0.linear_attn.in_proj_qkv",
        "model.language_model.layers.0.linear_attn.in_proj_z",
        "model.language_model.layers.0.linear_attn.in_proj_b",
        "model.language_model.layers.0.linear_attn.in_proj_a",
        "model.language_model.layers.0.linear_attn.out_proj",
        "model.language_model.layers.3.self_attn.q_proj",
        "model.language_model.layers.3.self_attn.k_proj",
        "model.language_model.layers.3.self_attn.v_proj",
        "model.language_model.layers.3.self_attn.o_proj",
        "model.language_model.layers.3.mlp.gate_proj",
        "model.language_model.layers.3.mlp.up_proj",
        "model.language_model.layers.3.mlp.down_proj",
    ]
    must_not_match = [
        "model.visual.blocks.0.attn.qkv",
        "model.visual.blocks.0.attn.proj",
        "model.language_model.embed_tokens",
        "model.language_model.layers.3.input_layernorm",
    ]
    assert all(pattern.fullmatch(name) for name in must_match)
    assert not any(pattern.fullmatch(name) for name in must_not_match)


def test_dtype_mapping_follows_the_device():
    assert resolve_torch_dtype(RunConfig(dtype="auto"), "mps") is torch.float32
    assert resolve_torch_dtype(RunConfig(dtype="auto"), "cuda") is torch.bfloat16
    assert resolve_torch_dtype(RunConfig(dtype="fp16"), "mps") is torch.float16


def test_device_resolution_uses_a_torch_device():
    device = resolve_torch_device(RunConfig(device="cpu"))
    assert device.type == "cpu"


def test_generation_kwargs_stop_at_im_end(stub_tokenizer):
    kwargs = generation_kwargs(stub_tokenizer, RunConfig(max_new_tokens=32))
    assert kwargs["max_new_tokens"] == 32
    assert kwargs["eos_token_id"] == stub_tokenizer.eos_token_id
    assert kwargs["pad_token_id"] == stub_tokenizer.pad_token_id
    assert kwargs["do_sample"] is False


def test_count_trainable_parameters_reports_a_percentage():
    import torch.nn as nn

    model = nn.Linear(4, 2)
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.bias.requires_grad = True
    summary = count_trainable_parameters(model)
    assert "trainable" in summary and "%" in summary


def _tiny_config():
    from transformers import Qwen3_5Config

    return Qwen3_5Config(
        text_config=dict(
            num_hidden_layers=1,
            layer_types=["full_attention"],
            hidden_size=16,
            num_attention_heads=1,
            num_key_value_heads=1,
            head_dim=8,
            intermediate_size=32,
            linear_num_key_heads=1,
            linear_num_value_heads=1,
            linear_key_head_dim=8,
            linear_value_head_dim=8,
            max_position_embeddings=128,
            mtp_num_hidden_layers=0,
            vocab_size=1024,
        ),
        vision_config=dict(
            depth=1,
            hidden_size=16,
            intermediate_size=16,
            num_heads=1,
            out_hidden_size=16,
            num_position_embeddings=4,
            patch_size=16,
        ),
    )


def test_load_model_reads_a_local_checkpoint_without_deprecation_noise(tmp_path, caplog):
    import logging

    from transformers import Qwen3_5ForConditionalGeneration

    from model_utils import load_model

    checkpoint = tmp_path / "tiny"
    Qwen3_5ForConditionalGeneration(_tiny_config()).save_pretrained(checkpoint)

    with caplog.at_level(logging.WARNING):
        model = load_model(RunConfig(model_id=str(checkpoint), device="cpu"), for_training=True)

    assert next(model.parameters()).dtype is torch.float32
    assert next(model.parameters()).device.type == "cpu"
    assert model.config.use_cache is False
    assert "torch_dtype" not in caplog.text


def test_incompatible_transformers_peft_pair_gives_an_actionable_error(monkeypatch):
    import model_utils

    def broken(model, config):
        raise RecursionError("maximum recursion depth exceeded")

    monkeypatch.setattr(model_utils, "get_peft_model", broken)
    with pytest.raises(RuntimeError, match=r"transformers .* peft .*requirements\.txt"):
        model_utils.wrap_with_lora(object(), RunConfig())
