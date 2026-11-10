import pytest

torch = pytest.importorskip("torch")

from config import RunConfig
from conftest import SavableStubTokenizer
import merge_adapter


def _config(tiny, **overrides):
    return RunConfig(model_id=str(tiny["base_dir"]), output_dir=tiny["out_dir"], device="cpu", **overrides)


@pytest.fixture(autouse=True)
def stub_tokenizer_loader(monkeypatch):
    monkeypatch.setattr(merge_adapter, "load_tokenizer", lambda config: SavableStubTokenizer())


def test_merged_model_path_defaults_next_to_the_adapter():
    config = RunConfig()
    assert config.merged_model_path == config.output_dir / "merged_model_CTI"


def test_merged_weights_reproduce_the_adapter_outputs(tiny_adapter):
    from transformers import AutoModelForImageTextToText

    config = _config(tiny_adapter)
    path = merge_adapter.merge_adapter(config)

    merged = AutoModelForImageTextToText.from_pretrained(path, dtype=torch.float32).eval()
    peft_model = tiny_adapter["peft_model"].eval()
    input_ids = torch.tensor([[3, 14, 15, 92, 6]])
    with torch.no_grad():
        expected = peft_model(input_ids=input_ids).logits
        actual = merged(input_ids=input_ids).logits
    assert torch.allclose(expected, actual, atol=1e-5)


def test_merged_checkpoint_is_a_plain_model_with_no_lora_leftovers(tiny_adapter):
    from safetensors import safe_open

    path = merge_adapter.merge_adapter(_config(tiny_adapter))
    keys = []
    for shard in path.glob("*.safetensors"):
        with safe_open(shard, "pt") as handle:
            keys.extend(handle.keys())
    assert keys and not any("lora" in key for key in keys)
    assert not (path / "adapter_config.json").exists()
    assert (path / "config.json").exists()
    assert (path / "tokenizer_config.json").exists()


def test_missing_adapter_directory_names_the_path(tiny_adapter, tmp_path):
    config = RunConfig(model_id=str(tiny_adapter["base_dir"]), output_dir=tmp_path / "empty", device="cpu")
    with pytest.raises(FileNotFoundError, match="peft_model_CTI"):
        merge_adapter.merge_adapter(config)


def test_existing_output_is_not_overwritten_without_the_flag(tiny_adapter):
    config = _config(tiny_adapter)
    config.merged_model_path.mkdir(parents=True)
    (config.merged_model_path / "keep.txt").write_text("precious")

    with pytest.raises(FileExistsError, match="merged_model_CTI"):
        merge_adapter.merge_adapter(config)
    assert (config.merged_model_path / "keep.txt").read_text() == "precious"

    merge_adapter.merge_adapter(_config(tiny_adapter, overwrite=True))
    assert not (config.merged_model_path / "keep.txt").exists()
    assert (config.merged_model_path / "config.json").exists()
