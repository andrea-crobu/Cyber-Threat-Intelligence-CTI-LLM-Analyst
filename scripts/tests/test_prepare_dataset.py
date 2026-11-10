import pandas as pd
import pytest
from datasets import Dataset

from config import RunConfig
from cti_preprocessing import CTIPreprocessor
from prepare_dataset import load_prepared, prepare_datasets, save_prepared, split_dataset


@pytest.fixture
def rows():
    return pd.DataFrame(
        {
            "text": [f"Report number {i} mentions DragonOK." for i in range(100)],
            "entities": [
                str([{"id": 0, "label": "threat-actor", "start_offset": 25, "end_offset": 33}])
                for _ in range(100)
            ],
            "relations": ["[]" for _ in range(100)],
            "diagnosis": [f"Diagnosis {i}." for i in range(100)],
        }
    )


def test_split_is_eighty_ten_ten(rows):
    splits = split_dataset(Dataset.from_pandas(rows), RunConfig())
    assert (len(splits["train"]), len(splits["validation"]), len(splits["test"])) == (80, 10, 10)


def test_split_rows_are_disjoint_and_complete(rows):
    splits = split_dataset(Dataset.from_pandas(rows), RunConfig())
    train = {row["text"] for row in splits["train"]}
    validation = {row["text"] for row in splits["validation"]}
    test = {row["text"] for row in splits["test"]}
    assert not (train & validation or train & test or validation & test)
    assert len(train | validation | test) == 100


def test_prepared_splits_have_matching_input_and_label_lengths(rows, stub_tokenizer):
    preprocessor = CTIPreprocessor(stub_tokenizer)
    prepared = prepare_datasets(rows, preprocessor, RunConfig())

    assert set(prepared) == {"train", "validation", "test"}
    for split in prepared.values():
        assert len(split["input_ids"]) == len(split["labels"])
        for input_ids, labels in zip(split["input_ids"], split["labels"]):
            assert len(input_ids) == len(labels)
            assert labels[-1] == stub_tokenizer.eos_token_id


def test_pickle_round_trip_preserves_token_ids(tmp_path, rows, stub_tokenizer):
    prepared = prepare_datasets(rows, CTIPreprocessor(stub_tokenizer), RunConfig())
    path = tmp_path / "dataset.pkl"
    save_prepared(prepared, path)
    assert load_prepared(path) == prepared


def test_missing_dataset_columns_raise_a_named_error(stub_tokenizer):
    preprocessor = CTIPreprocessor(stub_tokenizer)
    with pytest.raises(ValueError, match="entities"):
        prepare_datasets(pd.DataFrame({"text": ["only text"]}), preprocessor, RunConfig())


def test_split_ratios_come_from_the_config(rows):
    splits = split_dataset(Dataset.from_pandas(rows), RunConfig(train_size=0.5, validation_size=0.25, test_size=0.25))
    assert (len(splits["train"]), len(splits["validation"]), len(splits["test"])) == (50, 25, 25)
