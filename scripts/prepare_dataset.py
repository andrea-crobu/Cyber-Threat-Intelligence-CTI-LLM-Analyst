"""Phase 1: load the CTI dataset, split it 80/10/10, tokenize, and save the result."""

from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset

from config import RunConfig, parse_args
from cti_preprocessing import CTIPreprocessor


def split_dataset(dataset, config: RunConfig) -> DatasetDict:
    """80/10/10 with ``shuffle=False``, matching the notebook's two-step split."""
    remainder = config.validation_size + config.test_size
    first_split = dataset.train_test_split(
        train_size=config.train_size,
        test_size=remainder,
        shuffle=False,
    )
    validation_fraction = config.validation_size / remainder
    second_split = first_split["test"].train_test_split(
        train_size=validation_fraction,
        test_size=1 - validation_fraction,
        shuffle=False,
    )
    return DatasetDict(
        {
            "train": first_split["train"],
            "validation": second_split["train"],
            "test": second_split["test"],
        }
    )


def prepare_datasets(
    dataframe: pd.DataFrame, preprocessor: CTIPreprocessor, config: RunConfig
) -> dict[str, dict[str, list[list[int]]]]:
    """Tokenize every split into ``{"input_ids": [...], "labels": [...]}`` lists."""
    preprocessor.validate_dataframe(dataframe)
    splits = split_dataset(_as_dataset(dataframe), config)
    prepared = {}
    for name, split in splits.items():
        frame = split.to_pandas()
        prepared[name] = {
            "input_ids": preprocessor.preprocess_input_data(frame, for_training=True),
            "labels": preprocessor.preprocess_output_data(frame),
        }
    return prepared


def save_prepared(dataset_dict: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        pickle.dump(dataset_dict, handle)


def load_prepared(path: Path) -> dict:
    with open(path, "rb") as handle:
        return pickle.load(handle)


def _as_dataset(dataframe: pd.DataFrame) -> Dataset:
    return Dataset.from_pandas(dataframe, preserve_index=False)


def main(argv: list[str] | None = None) -> None:
    config = parse_args(argv)

    from model_utils import load_tokenizer

    tokenizer = load_tokenizer(config)
    preprocessor = CTIPreprocessor(tokenizer, max_length=config.max_length)

    dataset = load_dataset(config.dataset_name)
    split_name = "train" if "train" in dataset else list(dataset)[0]
    dataframe = dataset[split_name].to_pandas()
    print(f"Loaded {len(dataframe)} rows from {config.dataset_name} ({split_name})")

    prepared = prepare_datasets(dataframe, preprocessor, config)
    for name, split in prepared.items():
        print(f"{name}: {len(split['input_ids'])} examples")

    example = 0
    prompt_ids = prepared["train"]["input_ids"][example]
    label_ids = prepared["train"]["labels"][example]
    if len(prompt_ids) != len(label_ids):
        raise ValueError(
            f"Prompt and label lengths diverged: {len(prompt_ids)} != {len(label_ids)}"
        )
    print("\nPrompt sent to the model (chat framing kept):\n")
    print(preprocessor.decode(prompt_ids, preserve_special_tokens=True))
    print("\nExpected answer of the model:\n")
    print(preprocessor.decode(label_ids))

    save_prepared(prepared, config.dataset_path)
    print(f"\nSaved to {config.dataset_path}")


if __name__ == "__main__":
    main()
