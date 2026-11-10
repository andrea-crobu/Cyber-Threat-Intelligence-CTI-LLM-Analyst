"""Torch-side helpers: device and dtype policy, model loading, LoRA wrapping."""

from __future__ import annotations

import os

# The gated-delta-net fallback in Qwen3.5 calls torch.linalg.solve_triangular, which
# has no MPS kernel. Without this, the first forward pass fails on Apple silicon.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import RunConfig, resolve_device, resolve_dtype_name

IMAGE_TEXT_CLASS = "AutoModelForImageTextToText"


def resolve_torch_device(config: RunConfig) -> torch.device:
    name = resolve_device(
        config.device,
        mps_available=torch.backends.mps.is_available(),
        cuda_available=torch.cuda.is_available(),
    )
    return torch.device(name)


def resolve_torch_dtype(config: RunConfig, device: str) -> torch.dtype:
    return getattr(torch, resolve_dtype_name(config.dtype, device))


def load_tokenizer(config: RunConfig):
    tokenizer = AutoTokenizer.from_pretrained(config.model_id, token=config.hf_token())
    if tokenizer.pad_token_id is None:
        raise ValueError(f"{config.model_id} tokenizer has no pad token; refusing to reuse eos")
    return tokenizer


def load_model(config: RunConfig, for_training: bool = False):
    device = resolve_torch_device(config)
    dtype = resolve_torch_dtype(config, device.type)
    common = {
        "dtype": dtype,
        "token": config.hf_token(),
        "low_cpu_mem_usage": True,
        "attn_implementation": "sdpa",
    }

    import transformers

    if hasattr(transformers, IMAGE_TEXT_CLASS):
        loader = getattr(transformers, IMAGE_TEXT_CLASS)
        try:
            model = loader.from_pretrained(config.model_id, **common)
        except (ValueError, KeyError) as error:
            model = AutoModelForCausalLM.from_pretrained(config.model_id, **common)
    else:
        model = AutoModelForCausalLM.from_pretrained(config.model_id, **common)

    model.to(device)
    if for_training:
        model.config.use_cache = False
    return model


def build_lora_config(config: RunConfig) -> LoraConfig:
    return LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.lora_target_modules,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )


def wrap_with_lora(model, config: RunConfig):
    try:
        return get_peft_model(model, build_lora_config(config))
    except RecursionError as error:
        import peft
        import transformers

        raise RuntimeError(
            f"transformers {transformers.__version__} and peft {peft.__version__} are "
            "incompatible (infinite recursion while peft probes activation checkpointing). "
            "Reinstall the exact pair from requirements.txt: pip install -r scripts/requirements.txt "
            "--force-reinstall --no-deps transformers peft"
        ) from error


def count_trainable_parameters(model) -> str:
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    return (
        f"trainable model parameters: {trainable}\n"
        f"all model parameters: {total}\n"
        f"percentage of trainable model parameters: {100 * trainable / total:.2f}%"
    )


def generation_kwargs(tokenizer, config: RunConfig) -> dict:
    return {
        "max_new_tokens": config.max_new_tokens,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "do_sample": False,
    }
