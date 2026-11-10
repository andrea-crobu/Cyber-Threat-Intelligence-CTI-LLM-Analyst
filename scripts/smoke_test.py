"""One forward+backward step on two real examples, to validate new hardware.

Run this before train.py on a fresh machine: it proves the checkpoint loads, the
gated-delta-net fallback runs on this device, and it reports the step time so the
full run can be projected.
"""

from __future__ import annotations

import time

import torch
from torch.optim import AdamW

from config import RunConfig, parse_args
from model_utils import (
    count_trainable_parameters,
    load_model,
    load_tokenizer,
    resolve_torch_device,
    wrap_with_lora,
)
from prepare_dataset import load_prepared


def build_smoke_batch(prepared: dict) -> dict[str, list[list[int]]]:
    """The first two already-tokenized training examples."""
    return {
        "input_ids": prepared["train"]["input_ids"][:2],
        "labels": prepared["train"]["labels"][:2],
    }


def move_to_device(batch: dict, device: torch.device) -> dict:
    """The Trainer does this for train.py; a hand-rolled step has to do it itself."""
    return {key: value.to(device) for key, value in batch.items()}


def run_smoke_test(config: RunConfig) -> dict[str, float]:
    if not config.dataset_path.exists():
        raise FileNotFoundError(
            f"Prepared dataset not found at {config.dataset_path}; run prepare_dataset.py first"
        )

    from train import build_data_collator

    tokenizer = load_tokenizer(config)
    peft_model = wrap_with_lora(load_model(config, for_training=True), config)
    print(count_trainable_parameters(peft_model))

    batch = build_smoke_batch(load_prepared(config.dataset_path))
    features = build_data_collator(tokenizer)(
        [{"input_ids": ids, "labels": labels} for ids, labels in zip(batch["input_ids"], batch["labels"])]
    )

    parameters = [parameter for parameter in peft_model.parameters() if parameter.requires_grad]
    optimizer = AdamW(parameters, lr=config.learning_rate)
    peft_model.train()

    start = time.perf_counter()
    outputs = peft_model(**move_to_device(dict(features), resolve_torch_device(config)))
    outputs.loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    step_seconds = time.perf_counter() - start

    return {"step_seconds": step_seconds, "loss": float(outputs.loss.detach())}


def main(argv: list[str] | None = None) -> None:
    config = parse_args(argv)
    result = run_smoke_test(config)
    projected = result["step_seconds"] * 380
    print(f"\nstep time: {result['step_seconds']:.2f}s on {resolve_torch_device(config)}")
    print(f"loss: {result['loss']:.4f}")
    print(f"projected 380-step epoch: {projected / 60:.1f} min (plus evaluation)")
    if torch.backends.mps.is_available() and hasattr(torch, "mps"):
        print(f"mps allocated: {torch.mps.current_allocated_memory() / 1e9:.1f} GB")
    elif torch.cuda.is_available():
        print(f"cuda peak allocated: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
