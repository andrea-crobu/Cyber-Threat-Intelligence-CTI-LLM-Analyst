from pathlib import Path

import pytest

from config import RunConfig, build_arg_parser, parse_args, resolve_device, resolve_dtype_name


def test_defaults_match_the_project():
    config = RunConfig()
    assert config.model_id == "Qwen/Qwen3.5-4B"
    assert (config.train_size, config.validation_size, config.test_size) == (0.8, 0.1, 0.1)
    assert (config.lora_r, config.lora_alpha, config.lora_dropout) == (32, 32, 0.05)
    assert (config.per_device_train_batch_size, config.num_train_epochs) == (1, 1.0)
    assert config.learning_rate == pytest.approx(1e-3)


def test_artifact_paths_join_the_configured_dirs():
    config = RunConfig(data_dir=Path("/tmp/d"), output_dir=Path("/tmp/o"))
    assert config.dataset_path == Path("/tmp/d/dataset_CTI_qwen3.5-4B.pkl")
    assert config.peft_model_path == Path("/tmp/o/peft_model_CTI")
    assert config.predictions_path == Path("/tmp/o/model_predictions.csv")
    assert config.rouge_path == Path("/tmp/o/rouge_scores.json")


def test_hf_token_reads_only_the_configured_env_var(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert RunConfig().hf_token() is None
    monkeypatch.setenv("HF_TOKEN", "secret")
    assert RunConfig().hf_token() == "secret"
    monkeypatch.setenv("OTHER_TOKEN", "other")
    assert RunConfig(hf_token_env="OTHER_TOKEN").hf_token() == "other"


def test_auto_device_prefers_mps_then_cuda_then_cpu():
    assert resolve_device("auto", mps_available=True, cuda_available=True) == "mps"
    assert resolve_device("auto", mps_available=False, cuda_available=True) == "cuda"
    assert resolve_device("auto", mps_available=False, cuda_available=False) == "cpu"
    assert resolve_device("cuda", mps_available=True, cuda_available=False) == "cuda"


def test_unknown_device_is_rejected():
    with pytest.raises(ValueError, match="device"):
        resolve_device("tpu", mps_available=False, cuda_available=False)


def test_auto_dtype_is_fp32_on_mps_and_cpu_and_bf16_on_cuda():
    assert resolve_dtype_name("auto", "mps") == "float32"
    assert resolve_dtype_name("auto", "cpu") == "float32"
    assert resolve_dtype_name("auto", "cuda") == "bfloat16"


def test_bfloat16_is_rejected_on_mps():
    with pytest.raises(ValueError, match="bfloat16"):
        resolve_dtype_name("bf16", "mps")


def test_flags_override_defaults():
    config = parse_args(["--train-size", "0.9", "--model-id", "other/model", "--max-length", "512"])
    assert config.train_size == 0.9
    assert config.model_id == "other/model"
    assert config.max_length == 512


def test_parsing_starting_from_defaults_keeps_untouched_fields():
    config = parse_args(["--seed", "7"], defaults=RunConfig(output_dir=Path("/tmp/o")))
    assert config.seed == 7
    assert config.output_dir == Path("/tmp/o")


def test_parser_has_a_flag_for_every_dataclass_field():
    parser = build_arg_parser()
    options = {action.dest for action in parser._actions}
    assert {f.name for f in RunConfig.__dataclass_fields__.values()} <= options


def test_unset_flags_leave_no_trace_in_the_namespace():
    namespace = build_arg_parser().parse_args([])
    assert not hasattr(namespace, "train_size")
    assert not hasattr(namespace, "skip_rouge")
