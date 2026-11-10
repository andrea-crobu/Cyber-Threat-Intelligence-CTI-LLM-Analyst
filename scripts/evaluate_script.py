"""Phase 3: generate with the base and fine-tuned model, then score with ROUGE.

Named ``evaluate_script`` because a module called ``evaluate`` in this directory
would shadow the Hugging Face ``evaluate`` package this file depends on.
``run_evaluation.py`` is the thin entry point.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch

from config import RunConfig, parse_args
from model_utils import (
    generation_kwargs,
    load_model,
    load_tokenizer,
    resolve_torch_device,
)
from prepare_dataset import load_prepared

IGNORE_INDEX = -100
MODEL_COLUMNS = ("base_model", "fine_tuned_model")


def generate_predictions(
    base_model,
    peft_model,
    tokenizer,
    inputs: list[list[int]],
    labels: list[list[int]],
    config: RunConfig,
) -> pd.DataFrame:
    """Greedy-decode both models on every test prompt, keeping the ground truth."""
    kwargs = generation_kwargs(tokenizer, config)
    rows = []
    for index, (input_ids, label_ids) in enumerate(zip(inputs, labels)):
        ground_truth = tokenizer.decode(
            [token for token in label_ids if token != IGNORE_INDEX], skip_special_tokens=True
        )
        rows.append(
            {
                "ground_truth": ground_truth,
                "base_model": _generate_text(base_model, tokenizer, input_ids, kwargs),
                "fine_tuned_model": _generate_text(peft_model, tokenizer, input_ids, kwargs),
            }
        )
        print(f"[{index + 1}/{len(inputs)}] generated")
    return pd.DataFrame(rows, columns=["ground_truth", *MODEL_COLUMNS])


def compute_rouge(predictions: pd.DataFrame) -> dict[str, dict[str, float]]:
    rouge = _load_rouge()
    references = predictions["ground_truth"].tolist()
    return {
        column: rouge.compute(
            predictions=predictions[column].tolist(),
            references=references,
            use_aggregator=True,
            use_stemmer=True,
        )
        for column in MODEL_COLUMNS
    }


def format_rouge_table(scores: dict[str, dict[str, float]]) -> str:
    metrics = list(scores["base_model"].keys())
    header = f"{'metric':<10}" + "".join(f"{column:>18}" for column in scores)
    lines = [header, "-" * len(header)]
    for metric in metrics:
        row = f"{metric:<10}" + "".join(f"{scores[column][metric]:>18.4f}" for column in scores)
        lines.append(row)
    return "\n".join(lines)


def _generate_text(model, tokenizer, input_ids: list[int], kwargs: dict) -> str:
    output = model.generate(input_ids=torch.tensor([input_ids]), **kwargs)
    continuation = output[0][len(input_ids) :].tolist()
    return tokenizer.decode(continuation, skip_special_tokens=True)


def _load_rouge():
    import evaluate as evaluate_library

    return evaluate_library.load("rouge")


class _BaseModelView:
    """Expose the LoRA model with its adapters switched off, without a second copy."""

    def __init__(self, peft_model):
        self._peft_model = peft_model

    def generate(self, **kwargs):
        with self._peft_model.disable_adapter():
            return self._peft_model.generate(**kwargs)


def main(argv: list[str] | None = None) -> None:
    config = parse_args(argv)

    if not config.peft_model_path.is_dir():
        raise FileNotFoundError(
            f"Adapters not found at {config.peft_model_path}; run train.py first"
        )
    if not config.dataset_path.exists():
        raise FileNotFoundError(
            f"Prepared dataset not found at {config.dataset_path}; run prepare_dataset.py first"
        )

    from peft import PeftModel

    tokenizer = load_tokenizer(config)
    prepared = load_prepared(config.dataset_path)

    base_model = load_model(config)
    peft_model = PeftModel.from_pretrained(
        base_model, config.peft_model_path, is_trainable=False
    ).to(resolve_torch_device(config))
    peft_model.eval()

    predictions = generate_predictions(
        _BaseModelView(peft_model),
        peft_model,
        tokenizer,
        inputs=prepared["test"]["input_ids"],
        labels=prepared["test"]["labels"],
        config=config,
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(config.predictions_path, index=False)
    print(f"Predictions saved to {config.predictions_path}")

    if config.skip_rouge:
        return

    scores = compute_rouge(predictions)
    print("\n" + format_rouge_table(scores))
    Path(config.rouge_path).write_text(json.dumps(scores, indent=2))
    print(f"ROUGE scores saved to {config.rouge_path}")


if __name__ == "__main__":
    main()
