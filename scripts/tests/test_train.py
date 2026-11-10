import pytest

transformers = pytest.importorskip("transformers")

from config import RunConfig
from train import build_training_arguments


def test_training_arguments_keep_notebook_parity(tmp_path):
    arguments = build_training_arguments(RunConfig(output_dir=tmp_path))
    assert arguments.per_device_train_batch_size == 1
    assert arguments.per_device_eval_batch_size == 1
    assert arguments.num_train_epochs == 1.0
    assert arguments.eval_steps == 100
    assert arguments.logging_steps == 100
    assert arguments.seed == 42
    assert arguments.report_to == []
    assert arguments.output_dir == str(tmp_path / "training")


def test_training_arguments_never_request_bf16(tmp_path):
    arguments = build_training_arguments(RunConfig(output_dir=tmp_path))
    assert arguments.bf16 is False
    assert arguments.fp16 is False


def test_collator_pads_labels_with_the_ignore_index(stub_tokenizer):
    from train import build_data_collator

    collator = build_data_collator(stub_tokenizer)
    assert collator.label_pad_token_id == -100
