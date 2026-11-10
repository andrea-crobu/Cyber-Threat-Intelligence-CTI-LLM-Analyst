import pandas as pd
import pytest

from config import RunConfig
from evaluate_script import compute_rouge, format_rouge_table, generate_predictions


def test_missing_adapter_directory_names_the_path(tmp_path):
    from evaluate_script import main

    with pytest.raises(FileNotFoundError, match="absent"):
        main(["--output-dir", str(tmp_path), "--peft-dirname", "absent", "--skip-rouge"])


def test_rouge_table_prints_every_metric():
    scores = {
        "base_model": {"rouge1": 0.28, "rouge2": 0.11, "rougeL": 0.19, "rougeLsum": 0.26},
        "fine_tuned_model": {"rouge1": 0.65, "rouge2": 0.40, "rougeL": 0.5, "rougeLsum": 0.57},
    }
    table = format_rouge_table(scores)
    for metric in ("rouge1", "rouge2", "rougeL", "rougeLsum"):
        assert metric in table
    assert "0.65" in table


def test_predictions_are_trimmed_to_the_generated_tokens():
    torch = pytest.importorskip("torch")

    class FakeModel:
        device = torch.device("cpu")

        def __init__(self, token):
            self.token = token

        def generate(self, input_ids, **kwargs):
            continuation = torch.tensor([[self.token, self.token]])
            return torch.cat([input_ids, continuation], dim=1)

    class DecodingTokenizer:
        eos_token_id = 1
        pad_token_id = 0

        def decode(self, ids, skip_special_tokens=False):
            return f"ids={list(ids)}"

    frame = generate_predictions(
        FakeModel(5),
        FakeModel(7),
        DecodingTokenizer(),
        inputs=[[1, 2, 3]],
        labels=[[-100, -100, 4]],
        config=RunConfig(),
    )
    assert frame.loc[0, "base_model"] == "ids=[5, 5]"
    assert frame.loc[0, "fine_tuned_model"] == "ids=[7, 7]"
    assert frame.loc[0, "ground_truth"] == "ids=[4]"


def test_compute_rouge_scores_both_models(monkeypatch):
    import evaluate_script

    calls = []

    class FakeRouge:
        def compute(self, predictions, references, use_aggregator, use_stemmer):
            calls.append(predictions[0])
            return {"rouge1": 0.5}

    monkeypatch.setattr(evaluate_script, "_load_rouge", lambda: FakeRouge())
    frame = pd.DataFrame(
        {
            "ground_truth": ["a", "b"],
            "base_model": ["a", "b"],
            "fine_tuned_model": ["c", "d"],
        }
    )
    scores = compute_rouge(frame)
    assert set(scores) == {"base_model", "fine_tuned_model"}
    assert calls == ["a", "c"]


def test_no_module_named_evaluate_shadows_the_rouge_package():
    """A scripts/evaluate.py would win the import and break ROUGE scoring."""
    import evaluate as evaluate_package

    scripts_dir = __import__("pathlib").Path(__file__).resolve().parents[1]
    assert not (scripts_dir / "evaluate.py").exists()
    assert hasattr(evaluate_package, "load")


def test_generation_inputs_are_placed_on_the_models_device():
    torch = pytest.importorskip("torch")
    from evaluate_script import _BaseModelView, _generate_text

    seen = []

    class DeviceModel:
        device = torch.device("meta")

        def generate(self, input_ids, **kwargs):
            seen.append(input_ids.device.type)
            return torch.zeros((1, input_ids.shape[1] + 1), dtype=torch.long)

    class Tok:
        def decode(self, ids, skip_special_tokens=False):
            return "x"

    _generate_text(DeviceModel(), Tok(), [1, 2, 3], {})
    assert seen == ["meta"]

    class FakePeft(DeviceModel):
        def disable_adapter(self):
            import contextlib

            return contextlib.nullcontext()

    _generate_text(_BaseModelView(FakePeft()), Tok(), [1, 2, 3], {})
    assert seen == ["meta", "meta"]
