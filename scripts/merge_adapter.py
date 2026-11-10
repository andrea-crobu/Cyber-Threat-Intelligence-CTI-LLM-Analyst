"""Fold the trained LoRA adapter into the base weights and save a standalone model.

The result loads with plain ``from_pretrained`` (no peft needed), so it can be
served, converted to GGUF/MLX, or pushed to the Hub as an ordinary checkpoint.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from config import RunConfig, parse_args
from model_utils import load_model, load_tokenizer, resolve_torch_device


def merge_adapter(config: RunConfig) -> Path:
    adapter_path = config.peft_model_path
    if not adapter_path.is_dir():
        raise FileNotFoundError(f"Adapters not found at {adapter_path}; run train.py first")

    target = config.merged_model_path
    if target.exists():
        if not config.overwrite:
            raise FileExistsError(
                f"{target} already exists; pass --overwrite to replace it or --merged-dirname to use another name"
            )
        shutil.rmtree(target)

    from peft import PeftModel

    base_model = load_model(config)
    peft_model = PeftModel.from_pretrained(base_model, adapter_path, is_trainable=False)
    merged = peft_model.merge_and_unload()
    merged.to(resolve_torch_device(config))

    target.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(target)
    load_tokenizer(config).save_pretrained(target)
    print(f"Merged model saved to {target}")
    return target


def main(argv: list[str] | None = None) -> None:
    merge_adapter(parse_args(argv))


if __name__ == "__main__":
    main()
