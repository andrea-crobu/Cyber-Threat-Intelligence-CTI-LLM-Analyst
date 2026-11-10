"""Run configuration shared by every script in this folder."""

from __future__ import annotations

import argparse
import dataclasses
import os
import types
import typing
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_MODEL_ID = "Qwen/Qwen3.5-4B"
DEFAULT_DATASET_NAME = "swaption2009/cyber-threat-intelligence-custom-data"

# Text-decoder projections only: the 8 full-attention layers use q/k/v/o_proj, the
# 24 gated-delta-net layers use in_proj_*/out_proj, and every layer has an MLP.
# The vision tower is deliberately excluded - this task is text-only.
DEFAULT_LORA_TARGET_MODULES = (
    r".*language_model\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj"
    r"|in_proj_qkv|in_proj_z|in_proj_b|in_proj_a|out_proj)$"
)

_DEVICES = ("auto", "mps", "cuda", "cpu")
_DTYPE_ALIASES = {
    "auto": "auto",
    "fp32": "float32",
    "float32": "float32",
    "fp16": "float16",
    "float16": "float16",
    "bf16": "bfloat16",
    "bfloat16": "bfloat16",
}


@dataclass
class RunConfig:
    """Every knob the pipeline exposes, defaulted to the project's own settings."""

    model_id: str = DEFAULT_MODEL_ID
    dataset_name: str = DEFAULT_DATASET_NAME

    data_dir: Path = Path("data")
    output_dir: Path = Path("outputs")
    dataset_filename: str = "dataset_CTI_qwen3.5-4B.pkl"
    peft_dirname: str = "peft_model_CTI"
    predictions_filename: str = "model_predictions.csv"
    rouge_filename: str = "rouge_scores.json"

    train_size: float = 0.8
    validation_size: float = 0.1
    test_size: float = 0.1
    max_length: int | None = None
    seed: int = 42

    lora_r: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: str = DEFAULT_LORA_TARGET_MODULES

    learning_rate: float = 1e-3
    num_train_epochs: float = 1.0
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    eval_steps: int = 100
    logging_steps: int = 100

    max_new_tokens: int = 400
    skip_rouge: bool = False
    device: str = "auto"
    dtype: str = "auto"
    hf_token_env: str = "HF_TOKEN"

    @property
    def dataset_path(self) -> Path:
        return self.data_dir / self.dataset_filename

    @property
    def peft_model_path(self) -> Path:
        return self.output_dir / self.peft_dirname

    @property
    def predictions_path(self) -> Path:
        return self.output_dir / self.predictions_filename

    @property
    def rouge_path(self) -> Path:
        return self.output_dir / self.rouge_filename

    def hf_token(self) -> str | None:
        return os.environ.get(self.hf_token_env) or None


def resolve_device(requested: str, *, mps_available: bool, cuda_available: bool) -> str:
    """Turn ``--device`` into a concrete device, preferring Apple silicon over CUDA."""
    if requested not in _DEVICES:
        raise ValueError(f"Unknown device {requested!r}; expected one of {_DEVICES}")
    if requested != "auto":
        return requested
    if mps_available:
        return "mps"
    if cuda_available:
        return "cuda"
    return "cpu"


def resolve_dtype_name(requested: str, device: str) -> str:
    """Turn ``--dtype`` into a torch dtype name that the device can actually run."""
    if requested not in _DTYPE_ALIASES:
        raise ValueError(f"Unknown dtype {requested!r}; expected one of {tuple(_DTYPE_ALIASES)}")
    name = _DTYPE_ALIASES[requested]
    if name == "bfloat16" and device in ("mps", "cpu"):
        raise ValueError(
            f"bfloat16 is not supported on {device}; use --dtype fp32 (or fp16 on mps)"
        )
    if name == "auto":
        return "bfloat16" if device == "cuda" else "float32"
    return name


def _unwrap(annotation: object) -> object:
    """Return the concrete arm of ``X | None`` annotations."""
    origin = typing.get_origin(annotation)
    if origin in (typing.Union, types.UnionType):
        arms = [arm for arm in typing.get_args(annotation) if arm is not type(None)]
        if len(arms) == 1:
            return arms[0]
    return annotation


def build_arg_parser(defaults: RunConfig | None = None) -> argparse.ArgumentParser:
    """One ``--flag`` per dataclass field, each defaulting to ``None`` so overrides are detectable."""
    defaults = defaults or RunConfig()
    parser = argparse.ArgumentParser(
        description=f"Cyber Threat Intelligence pipeline on {defaults.model_id}",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    hints = typing.get_type_hints(RunConfig)
    for name in RunConfig.__dataclass_fields__:
        flag = "--" + name.replace("_", "-")
        annotation = _unwrap(hints.get(name, str))
        current = getattr(defaults, name)
        if isinstance(current, bool):
            parser.add_argument(
                flag,
                action="store_true",
                default=argparse.SUPPRESS,
                help=f"default: {current}",
            )
        else:
            parser.add_argument(
                flag,
                type=Path if annotation is Path else annotation,
                default=argparse.SUPPRESS,
                help=f"default: {current}",
            )
    return parser


def parse_args(argv: list[str] | None = None, defaults: RunConfig | None = None) -> RunConfig:
    """Parse CLI flags over a base config, applying only the flags that were given."""
    defaults = defaults or RunConfig()
    namespace = build_arg_parser(defaults).parse_args(argv)
    return dataclasses.replace(defaults, **vars(namespace))
