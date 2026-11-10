# scripts/ — CTI on Qwen3.5-4B Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the three-notebook CTI pipeline (dataset prep → LoRA fine-tune → ROUGE evaluation) as self-contained scripts in `scripts/`, with `Qwen/Qwen3.5-4B` instead of Llama-3.2-3B-Instruct, on an Apple silicon Mac.

**Architecture:** A small `scripts/` module set: one dataclass config, one vendored preprocessing module that re-frames the project prompt with Qwen's chat template, one torch helper module, and three thin entry points plus a smoke test. Nothing imports `Pipeline/`.

**Tech Stack:** Python 3.11+, torch (MPS), transformers @ main (4.57+, required by `qwen3_5`), peft, datasets, evaluate/rouge_score, pandas, pytest.

**Spec:** `docs/superpowers/specs/2026-11-10-qwen35-cti-scripts-design.md`

## Global Constraints

- Base model id default: `Qwen/Qwen3.5-4B` (`Qwen3_5ForConditionalGeneration`, vision-language; `transformers` from `main` required; `AutoModelForCausalLM` fails).
- Prompt framing: `apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True, enable_thinking=False)`.
- Tokenization parity: prompt ids and output ids tokenized separately, concatenated; labels = `[-100] * len(prompt_ids) + output_ids + eos_ids`.
- `eos = <|im_end|>`, `pad = <|endoftext|>`, `add_bos_token: false`. Never use the notebook's `pad_token = eos_token` line.
- Training defaults: batch 1, 1 epoch, `AdamW(lr=1e-3)` on LoRA params only, `eval_steps=100`, `logging_steps=100`, seed 42, `DataCollatorForSeq2Seq(label_pad_token_id=-100)`.
- LoRA: `r=32`, `alpha=32`, `dropout=0.05`, `task_type=CAUSAL_LM`, targets the text-decoder linears by regex; vision tower excluded.
- MPS: no bfloat16, no bitsandbytes; `attn_implementation="sdpa"`; `PYTORCH_ENABLE_MPS_FALLBACK=1` set before torch is imported.
- Secrets: HF token only from `HF_TOKEN`; no placeholder tokens in files.
- Artifacts: `<data_dir>/dataset_CTI_qwen3.5-4B.pkl`, `<output_dir>/peft_model_CTI/`, `<output_dir>/model_predictions.csv`, `<output_dir>/rouge_scores.json`.
- **No git commits.** Repo policy: leave everything in the working tree and tell the user what to commit.
- Python style: type hints on public functions, docstrings only where the contract is not obvious, no comments restating code.

## Review Focus

- **A 476-row dataset split three ways.** A user expects 380/48/48 with no row appearing in two splits. Task 4's tests pin sizes and disjointness.
- **Mask alignment.** A user expects the number of `-100` entries to equal the prompt token count exactly, so no prompt token contributes to loss and no output token is masked. Task 2 pins it, including for the chat-templated prompt.
- **Empty/malformed `entities`/`relations` cells.** The dataset stores them as Python-literal strings; a user expects a row with `"[]"` to produce an empty list rather than crash or emit `"nan"`. Task 2 pins it.
- **Truncation.** A user expects `--max-length` to cap each side, and the default (no cap) to change nothing relative to the notebooks. Task 2 pins both, and Task 4 pins that prompt and labels stay equal length after truncation.
- **A missing or unreadable checkpoint directory at evaluation time.** A user expects a clear error naming the directory, not a `FileNotFoundError` from deep inside `peft`. Task 6 pins it.

---

### Task 1: Configuration and device/dtype policy

**Files:**
- Create: `scripts/config.py`
- Test: `scripts/tests/test_config.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `DEFAULT_MODEL_ID: str`, `DEFAULT_DATASET_NAME: str`, `DEFAULT_LORA_TARGET_MODULES: str`
  - `@dataclass RunConfig` with fields: `model_id: str = DEFAULT_MODEL_ID`, `dataset_name: str = DEFAULT_DATASET_NAME`, `data_dir: Path = Path("data")`, `output_dir: Path = Path("outputs")`, `dataset_filename: str = "dataset_CTI_qwen3.5-4B.pkl"`, `peft_dirname: str = "peft_model_CTI"`, `predictions_filename: str = "model_predictions.csv"`, `rouge_filename: str = "rouge_scores.json"`, `train_size: float = 0.8`, `validation_size: float = 0.1`, `test_size: float = 0.1`, `max_length: int | None = None`, `seed: int = 42`, `lora_r: int = 32`, `lora_alpha: int = 32`, `lora_dropout: float = 0.05`, `lora_target_modules: str = DEFAULT_LORA_TARGET_MODULES`, `learning_rate: float = 1e-3`, `num_train_epochs: float = 1.0`, `per_device_train_batch_size: int = 1`, `per_device_eval_batch_size: int = 1`, `eval_steps: int = 100`, `logging_steps: int = 100`, `max_new_tokens: int = 400`, `device: str = "auto"`, `dtype: str = "auto"`, `hf_token_env: str = "HF_TOKEN"`
  - properties `dataset_path`, `peft_model_path`, `predictions_path`, `rouge_path` → `Path`
  - `RunConfig.hf_token() -> str | None` (reads `os.environ[self.hf_token_env]`, `None` if unset/empty)
  - `resolve_device(requested: str, *, mps_available: bool, cuda_available: bool) -> str`
  - `resolve_dtype_name(requested: str, device: str) -> str` → one of `"float32"`, `"float16"`, `"bfloat16"`
  - `build_arg_parser(defaults: RunConfig | None = None) -> argparse.ArgumentParser` (flags are `--` + field name with `_`→`-`; `Path` fields converted)
  - `parse_args(argv: list[str] | None, defaults: RunConfig | None = None) -> RunConfig`

- [ ] **Step 1: Write the failing test**

`scripts/tests/test_config.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd scripts && python -m pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'config'`

- [ ] **Step 3: Implement `scripts/config.py`**

Device/dtype logic stays torch-free so the test suite runs on machines without torch. `build_arg_parser` derives one flag per dataclass field, infers `type` from the field default (`Path` → `Path`, `int`/`float`/`str` as-is, `None` → `str`), and passes `default=None` so the parser only reports explicitly given flags; `parse_args` starts from `defaults or RunConfig()` and applies only the flags whose parsed value is not `None`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd scripts && python -m pytest tests/test_config.py -v`
Expected: PASS (10 passed)

---

### Task 2: Vendored preprocessing with chat-template framing

**Files:**
- Create: `scripts/cti_preprocessing.py`
- Test: `scripts/tests/test_preprocessing.py`, `scripts/tests/conftest.py`

**Interfaces:**
- Consumes: nothing from `Pipeline/`.
- Produces:
  - `PROMPT_INTRO: str` = `"You are a skilled AI Agent capable of doing CTI Analysis.\n\nGiven this threat report:\n"`
  - `PROMPT_OUTRO: str` = `"\n\nYou will extract the main entities and their relations; finally, you will generate a diagnosis of the threat.\n"`
  - `REQUIRED_COLUMNS: tuple[str, ...]` = `("text", "entities", "relations", "diagnosis")`
  - `class CTIPreprocessor(tokenizer, max_length: int | None = None, enable_thinking: bool = False)` with:
    - `build_prompt_text(self, report: str) -> str` — raw project prompt, intro + report + outro
    - `format_input_prompt(self, dataframe: pd.DataFrame) -> pd.DataFrame` — adds a `prompt` column holding the **chat-rendered** prompt
    - `process_entries_and_relations(self, row: pd.Series) -> pd.Series`
    - `format_output(self, dataframe: pd.DataFrame) -> pd.DataFrame` — adds an `output` column: `"\nEntities: …\nRelations: …\nDiagnosis: …"`
    - `tokenize(self, text: str) -> list[int]`
    - `eos_ids(self) -> list[int]`
    - `preprocess_input_data(self, dataframe: pd.DataFrame, for_training: bool = True) -> list[list[int]]`
    - `preprocess_output_data(self, dataframe: pd.DataFrame) -> list[list[int]]`
    - `decode(self, ids: list[int]) -> str` — drops `-100` before decoding

- [ ] **Step 1: Write the stub tokenizer**

`scripts/tests/conftest.py`:

```python
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class StubTokenizer:
    """Word-level stand-in for a HF tokenizer; records how it was called."""

    pad_token = "<|endoftext|>"
    eos_token = "<|im_end|>"
    pad_token_id = 0
    eos_token_id = 1
    model_max_length = 4096

    def __init__(self):
        self.vocab = {"<|im_end|>": 1, "<pad>": 0}
        self.template_calls = []
        self.encode_calls = []

    def _id(self, word):
        return self.vocab.setdefault(word, len(self.vocab))

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kwargs):
        self.template_calls.append(
            {"messages": messages, "tokenize": tokenize, "add_generation_prompt": add_generation_prompt, **kwargs}
        )
        text = f"<|im_start|>{messages[0]['role']}\n{messages[0]['content']}<|im_end|>\n<|im_start|>assistant\n"
        if kwargs.get("enable_thinking") is False:
            text += "<think>\n\n</think>\n\n"
        return text if not tokenize else [self._id(w) for w in text.split()]

    def __call__(self, text, truncation=False, max_length=None):
        self.encode_calls.append({"truncation": truncation, "max_length": max_length})
        ids = [self._id(word) for word in text.split()]
        if truncation and max_length is not None:
            ids = ids[:max_length]
        return SimpleNamespace(input_ids=ids)

    def decode(self, ids, skip_special_tokens=False):
        inverse = {value: key for key, value in self.vocab.items()}
        return " ".join(inverse.get(int(i), "?") for i in ids)


@pytest.fixture
def stub_tokenizer():
    return StubTokenizer()


@pytest.fixture
def report_frame():
    import pandas as pd

    return pd.DataFrame(
        [
            {
                "text": "NFlog and PoisonIvy were used by DragonOK.",
                "entities": str(
                    [
                        {"id": 0, "label": "malware", "start_offset": 0, "end_offset": 5},
                        {"id": 1, "label": "malware", "start_offset": 10, "end_offset": 19},
                        {"id": 2, "label": "threat-actor", "start_offset": 34, "end_offset": 42},
                    ]
                ),
                "relations": str(
                    [
                        {"from_id": 0, "to_id": 2, "type": "authored-by"},
                        {"from_id": 1, "to_id": 2, "type": "authored-by"},
                    ]
                ),
                "diagnosis": "DragonOK deploys malware.",
            },
            {
                "text": "Lazarus targeted banks.",
                "entities": str([{"id": 0, "label": "threat-actor", "start_offset": 0, "end_offset": 7}]),
                "relations": "[]",
                "diagnosis": "A threat actor targets finance.",
            },
        ]
    )
```

- [ ] **Step 2: Write the failing tests**

`scripts/tests/test_preprocessing.py`:

```python
import pandas as pd
import pytest

from cti_preprocessing import (
    PROMPT_INTRO,
    PROMPT_OUTRO,
    CTIPreprocessor,
)


def test_prompt_text_is_the_project_wording(stub_tokenizer):
    preprocessor = CTIPreprocessor(stub_tokenizer)
    assert preprocessor.build_prompt_text("Report body.") == (
        "You are a skilled AI Agent capable of doing CTI Analysis.\n\nGiven this threat report:\n"
        "Report body."
        "\n\nYou will extract the main entities and their relations; finally, you will generate a diagnosis of the threat.\n"
    )
    assert preprocessor.build_prompt_text("x") == f"{PROMPT_INTRO}x{PROMPT_OUTRO}"


def test_prompt_is_wrapped_once_in_the_chat_template_with_thinking_disabled(stub_tokenizer, report_frame):
    preprocessor = CTIPreprocessor(stub_tokenizer)
    prompts = preprocessor.format_input_prompt(report_frame)

    assert len(stub_tokenizer.template_calls) == len(report_frame)
    call = stub_tokenizer.template_calls[0]
    assert call["enable_thinking"] is False
    assert call["add_generation_prompt"] is True
    assert call["tokenize"] is False
    assert call["messages"] == [{"role": "user", "content": preprocessor.build_prompt_text(report_frame.loc[0, "text"])}]

    prompt = prompts.loc[0, "prompt"]
    assert prompt.startswith("<|im_start|>user\n")
    assert prompt.endswith("<|im_start|>assistant\n think\n\n</think>\n\n".replace(" think", " thinking"))
    assert "<|im_start|>user" not in prompt.split("<|im_start|>assistant")[0][1:]


def test_entities_and_relations_are_formatted_like_the_project(stub_tokenizer, report_frame):
    preprocessor = CTIPreprocessor(stub_tokenizer)
    row = preprocessor.process_entries_and_relations(report_frame.loc[0])
    assert row["entities"] == "NFlog (malware), PoisonIvy (malware), DragonOK (threat-actor)"
    assert row["relations"] == "NFlog to DragonOK (authored-by), PoisonIvy to DragonOK (authored-by)"


def test_empty_relation_list_yields_empty_string_not_nan(stub_tokenizer, report_frame):
    preprocessor = CTIPreprocessor(stub_tokenizer)
    row = preprocessor.process_entries_and_relations(report_frame.loc[1])
    assert row["relations"] == ""
    assert row["entities"] == "Lazarus (threat-actor)"


def test_output_string_has_the_three_sections(stub_tokenizer, report_frame):
    preprocessor = CTIPreprocessor(stub_tokenizer)
    output = preprocessor.format_output(report_frame).loc[0, "output"]
    assert output.startswith("\nEntities: NFlog (malware)")
    assert "\nRelations: NFlog to DragonOK (authored-by)" in output
    assert output.endswith("\nDiagnosis: DragonOK deploys malware.")


def test_training_inputs_append_the_masked_output_and_eos(stub_tokenizer, report_frame):
    preprocessor = CTIPreprocessor(stub_tokenizer)
    inputs = preprocessor.preprocess_input_data(report_frame, for_training=True)
    labels = preprocessor.preprocess_output_data(report_frame)

    assert len(inputs) == len(labels) == len(report_frame)
    assert [len(a) for a in inputs] == [len(b) for b in labels]
    assert inputs[0][-1] == stub_tokenizer.eos_token_id
    assert labels[0][-1] == stub_tokenizer.eos_token_id


def test_labels_mask_exactly_the_prompt_tokens(stub_tokenizer, report_frame):
    preprocessor = CTIPreprocessor(stub_tokenizer)
    prompt_ids = [stub_tokenizer(row) for row in preprocessor.format_input_prompt(report_frame)["prompt"]]
    labels = preprocessor.preprocess_output_data(report_frame)

    for prompt, label in zip(prompt_ids, labels):
        masked = [token for token in label if token == -100]
        assert len(masked) == len(prompt.input_ids)
        assert label[: len(prompt.input_ids)] == [-100] * len(prompt.input_ids)


def test_inference_inputs_are_prompt_only(stub_tokenizer, report_frame):
    preprocessor = CTIPreprocessor(stub_tokenizer)
    inputs = preprocessor.preprocess_input_data(report_frame, for_training=False)
    prompt = preprocessor.format_input_prompt(report_frame).loc[0, "prompt"]
    assert inputs[0] == stub_tokenizer(prompt).input_ids
    assert stub_tokenizer.eos_token_id not in inputs[0]


def test_max_length_truncates_each_side_and_keeps_lengths_equal(stub_tokenizer, report_frame):
    preprocessor = CTIPreprocessor(stub_tokenizer, max_length=5)
    inputs = preprocessor.preprocess_input_data(report_frame, for_training=True)
    labels = preprocessor.preprocess_output_data(report_frame)
    assert [len(a) for a in inputs] == [len(b) for b in labels]
    assert all(call["max_length"] == 5 for call in stub_tokenizer.encode_calls)


def test_no_max_length_means_no_truncation(stub_tokenizer, report_frame):
    preprocessor = CTIPreprocessor(stub_tokenizer)
    preprocessor.preprocess_input_data(report_frame)
    assert all(call["truncation"] is False for call in stub_tokenizer.encode_calls)


def test_missing_columns_raise_a_named_error(stub_tokenizer, report_frame):
    preprocessor = CTIPreprocessor(stub_tokenizer)
    with pytest.raises(ValueError, match="diagnosis"):
        preprocessor.preprocess_output_data(report_frame.drop(columns=["diagnosis"]))


def test_decode_drops_masked_tokens(stub_tokenizer, report_frame):
    preprocessor = CTIPreprocessor(stub_tokenizer)
    labels = preprocessor.preprocess_output_data(report_frame)
    decoded = preprocessor.decode(labels[0])
    assert "-100" not in decoded
    assert "Entities" in decoded or "<|im_end|>" in decoded
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd scripts && python -m pytest tests/test_preprocessing.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'cti_preprocessing'`

- [ ] **Step 4: Implement `scripts/cti_preprocessing.py`**

Port `process_entries_and_relations` and `format_output` from `Pipeline/preprocessing/CTI_Preprocessor.py` (same AST-literal parsing, same offset slicing, same separator strings), drop the `InputOutputPreprocessor` base class and its parameter-validation scaffolding, and route the prompt through `apply_chat_template`. Column validation lives in a private `_validate_columns(dataframe)` used by both `preprocess_*` methods.

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd scripts && python -m pytest tests/test_preprocessing.py -v`
Expected: PASS (12 passed)

---

### Task 3: Torch helpers — device, dtype, loaders, LoRA

**Files:**
- Create: `scripts/model_utils.py`
- Test: `scripts/tests/test_model_utils.py`

**Interfaces:**
- Consumes: `RunConfig`, `resolve_device`, `resolve_dtype_name` from Task 1.
- Produces:
  - `resolve_torch_device(config: RunConfig) -> torch.device`
  - `resolve_torch_dtype(config: RunConfig, device: str) -> torch.dtype`
  - `load_tokenizer(config: RunConfig)`
  - `load_model(config: RunConfig, for_training: bool = False)`
  - `build_lora_config(config: RunConfig) -> peft.LoraConfig`
  - `wrap_with_lora(model, config: RunConfig) -> peft.PeftModel`
  - `count_trainable_parameters(model) -> str`
  - `generation_kwargs(tokenizer, config: RunConfig) -> dict`

- [ ] **Step 1: Write the failing test**

`scripts/tests/test_model_utils.py` — pure-logic tests (LoRA config contents, generation kwargs, target regex) plus `pytest.importorskip("torch")` for the device dtype mapping:

```python
import pytest

torch = pytest.importorskip("torch")

from config import RunConfig
from model_utils import build_lora_config, count_trainable_parameters, generation_kwargs, resolve_torch_device, resolve_torch_dtype


def test_lora_config_matches_the_spec():
    lora = build_lora_config(RunConfig())
    assert (lora.r, lora.lora_alpha, lora.lora_dropout) == (32, 32, 0.05)
    assert lora.task_type == "CAUSAL_LM"
    assert lora.bias == "none"


def test_lora_target_regex_covers_text_decoder_projections_but_not_vision():
    import re

    pattern = re.compile(RunConfig().lora_target_modules)
    must_match = [
        "model.language_model.layers.0.linear_attn.in_proj_qkv",
        "model.language_model.layers.0.linear_attn.in_proj_z",
        "model.language_model.layers.0.linear_attn.in_proj_b",
        "model.language_model.layers.0.linear_attn.in_proj_a",
        "model.language_model.layers.0.linear_attn.out_proj",
        "model.language_model.layers.3.self_attn.q_proj",
        "model.language_model.layers.3.self_attn.k_proj",
        "model.language_model.layers.3.self_attn.v_proj",
        "model.language_model.layers.3.self_attn.o_proj",
        "model.language_model.layers.3.mlp.gate_proj",
        "model.language_model.layers.3.mlp.up_proj",
        "model.language_model.layers.3.mlp.down_proj",
    ]
    must_not_match = [
        "model.visual.blocks.0.attn.qkv",
        "model.visual.blocks.0.attn.proj",
        "model.language_model.embed_tokens",
        "model.language_model.layers.3.input_layernorm",
    ]
    assert all(pattern.fullmatch(name) for name in must_match)
    assert not any(pattern.fullmatch(name) for name in must_not_match)


def test_dtype_mapping_follows_the_device():
    assert resolve_torch_dtype(RunConfig(dtype="auto"), "mps") is torch.float32
    assert resolve_torch_dtype(RunConfig(dtype="auto"), "cuda") is torch.bfloat16
    assert resolve_torch_dtype(RunConfig(dtype="fp16"), "mps") is torch.float16


def test_device_resolution_uses_a_torch_device():
    device = resolve_torch_device(RunConfig(device="cpu"))
    assert device.type == "cpu"


def test_generation_kwargs_stop_at_im_end(stub_tokenizer):
    kwargs = generation_kwargs(stub_tokenizer, RunConfig(max_new_tokens=32))
    assert kwargs["max_new_tokens"] == 32
    assert kwargs["eos_token_id"] == stub_tokenizer.eos_token_id
    assert kwargs["pad_token_id"] == stub_tokenizer.pad_token_id
    assert kwargs["do_sample"] is False


def test_count_trainable_parameters_reports_a_percentage():
    import torch.nn as nn

    model = nn.Linear(4, 2)
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.bias.requires_grad = True
    summary = count_trainable_parameters(model)
    assert "trainable" in summary and "%" in summary
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd scripts && python -m pytest tests/test_model_utils.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'model_utils'` (or SKIPPED when torch is absent; install torch in the test venv so this task is actually exercised)

- [ ] **Step 3: Implement `scripts/model_utils.py`**

First two statements of the module (before `import torch`): `import os` and `os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")`. `load_model` tries `AutoModelForImageTextToText.from_pretrained(..., torch_dtype=..., attn_implementation="sdpa", token=...)` and falls back to `AutoModelForCausalLM` when the image-text class is unavailable (older transformers or a text-only checkpoint); it then moves the model to the resolved device and sets `config.use_cache = False` when `for_training`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd scripts && python -m pytest tests/test_model_utils.py -v`
Expected: PASS (6 passed), or SKIPPED rows if torch is absent — record which.

---

### Task 4: Dataset preparation entry point

**Files:**
- Create: `scripts/prepare_dataset.py`
- Test: `scripts/tests/test_prepare_dataset.py`

**Interfaces:**
- Consumes: `RunConfig` (Task 1), `CTIPreprocessor` (Task 2).
- Produces:
  - `split_dataset(dataset, config: RunConfig)` → `datasets.DatasetDict` with keys `train`, `validation`, `test` (80/10/10, `shuffle=False`, two-step split as in the notebook)
  - `prepare_datasets(dataset, preprocessor: CTIPreprocessor, config: RunConfig) -> dict[str, dict[str, list[list[int]]]]` — keys `train`/`validation`/`test`, each `{"input_ids": [...], "labels": [...]}`
  - `save_prepared(dataset_dict: dict, path: Path) -> None` (pickle)
  - `load_prepared(path: Path) -> dict` (pickle)
  - `main(argv: list[str] | None = None) -> None`

- [ ] **Step 1: Write the failing test**

`scripts/tests/test_prepare_dataset.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd scripts && python -m pytest tests/test_prepare_dataset.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'prepare_dataset'`

- [ ] **Step 3: Implement `scripts/prepare_dataset.py`**

`main` loads the dataset with `datasets.load_dataset(config.dataset_name)["train"]`, expands it to a DataFrame with `.to_pandas()`, builds the tokenizer via `model_utils.load_tokenizer`, prints the two sanity checks the notebook prints (decoded prompt, decoded expected output, and the `len(input) == len(labels)` assertion), then writes `config.dataset_path` (creating `config.data_dir`).

- [ ] **Step 4: Run test to verify it passes**

Run: `cd scripts && python -m pytest tests/test_prepare_dataset.py -v`
Expected: PASS (5 passed)

---

### Task 5: Training entry point

**Files:**
- Create: `scripts/train.py`
- Test: `scripts/tests/test_train.py`

**Interfaces:**
- Consumes: `RunConfig` (Task 1), `load_prepared` (Task 4), `model_utils` (Task 3).
- Produces:
  - `build_training_arguments(config: RunConfig) -> transformers.TrainingArguments`
  - `build_trainer(config: RunConfig, model, tokenizer, train_dataset, eval_dataset) -> transformers.Trainer`
  - `run_training(config: RunConfig) -> Path` — returns the adapter directory
  - `main(argv: list[str] | None = None) -> None`

- [ ] **Step 1: Write the failing test**

`scripts/tests/test_train.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd scripts && python -m pytest tests/test_train.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'train'`

- [ ] **Step 3: Implement `scripts/train.py`**

`build_trainer` uses `DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100)` (not the notebook's `model=peft_model` form, which refuses a vision-language config) and `optimizers=(torch.optim.AdamW(lora_parameters, lr=config.learning_rate), None)`. `run_training` loads the prepared pickle, wraps the lists in `datasets.Dataset.from_dict`, sets `model.config.use_cache = False`, trains, then saves the adapter and tokenizer to `config.peft_model_path`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd scripts && python -m pytest tests/test_train.py -v`
Expected: PASS (2 passed), or SKIPPED without transformers — record which.

---

### Task 6: Evaluation entry point

**Files:**
- Create: `scripts/evaluate.py`
- Test: `scripts/tests/test_evaluate.py`

**Interfaces:**
- Consumes: `RunConfig` (Task 1), `load_prepared` (Task 4), `model_utils` (Task 3).
- Produces:
  - `generate_predictions(base_model, peft_model, tokenizer, inputs: list[list[int]], labels: list[list[int]], config: RunConfig) -> pandas.DataFrame` — columns `ground_truth`, `base_model`, `fine_tuned_model`
  - `compute_rouge(predictions: pandas.DataFrame) -> dict[str, dict[str, float]]` — keys `base_model`, `fine_tuned_model`
  - `format_rouge_table(scores: dict) -> str`
  - `main(argv: list[str] | None = None) -> None`

- [ ] **Step 1: Write the failing test**

`scripts/tests/test_evaluate.py`:

```python
import pandas as pd
import pytest

from config import RunConfig
from evaluate_script import compute_rouge, format_rouge_table, generate_predictions


def test_missing_adapter_directory_names_the_path(tmp_path, monkeypatch):
    from evaluate_script import main

    config = tmp_path / "absent"
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
```

Note for the implementer: import the module as `evaluate_script` in tests — `scripts/evaluate.py` must therefore be importable under that alias, so add `scripts/evaluate_script.py` as the real module name and keep `scripts/evaluate.py` as a one-line entry point calling `evaluate_script.main()`. **Correction to the file list:** the implementation file is `scripts/evaluate_script.py` plus a thin `scripts/evaluate.py` wrapper, because `evaluate` collides with the installed `evaluate` package used for ROUGE.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd scripts && python -m pytest tests/test_evaluate.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'evaluate_script'`

- [ ] **Step 3: Implement `scripts/evaluate_script.py` and `scripts/evaluate.py`**

`main` loads the prepared pickle, checks `config.peft_model_path.is_dir()` and raises `FileNotFoundError(f"Adapters not found at {path}; run train.py first")`, loads base + PEFT, writes `config.predictions_path`, then ROUGE via a module-level `_load_rouge()` (returns `evaluate.load("rouge")`, importable for tests) and writes `config.rouge_path`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd scripts && python -m pytest tests/test_evaluate.py -v`
Expected: PASS (3 passed, 1 skipped without torch)

---

### Task 7: Smoke test for new hardware

**Files:**
- Create: `scripts/smoke_test.py`
- Test: `scripts/tests/test_smoke_test.py`

**Interfaces:**
- Consumes: Tasks 1–3.
- Produces:
  - `build_smoke_batch(preprocessor: CTIPreprocessor, dataframe, tokenizer) -> dict[str, list[list[int]]]`
  - `run_smoke_test(config: RunConfig) -> dict[str, float]` — keys `step_seconds`, `loss`
  - `main(argv: list[str] | None = None) -> None`

- [ ] **Step 1: Write the failing test**

`scripts/tests/test_smoke_test.py`:

```python
import pandas as pd
import pytest

from cti_preprocessing import CTIPreprocessor
from smoke_test import build_smoke_batch


def test_smoke_batch_holds_two_masked_examples(stub_tokenizer):
    frame = pd.DataFrame(
        {
            "text": ["Alpha used Beta.", "Gamma used Delta."],
            "entities": ["[]", "[]"],
            "relations": ["[]", "[]"],
            "diagnosis": ["One.", "Two."],
        }
    )
    batch = build_smoke_batch(CTIPreprocessor(stub_tokenizer), frame, stub_tokenizer)
    assert len(batch["input_ids"]) == 2
    assert len(batch["labels"]) == 2
    assert batch["labels"][0][0] == -100
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd scripts && python -m pytest tests/test_smoke_test.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'smoke_test'`

- [ ] **Step 3: Implement `scripts/smoke_test.py`**

`run_smoke_test` loads tokenizer and model, wraps with LoRA, builds the batch from two rows of the real dataset, runs one forward+backward with `torch.autocast` disabled, times it, and returns `{"step_seconds": ..., "loss": ...}`; `main` prints the numbers plus a projected full-run time for 380 steps.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd scripts && python -m pytest tests/test_smoke_test.py -v`
Expected: PASS (1 passed)

---

### Task 8: README, requirements, and artifact map

**Files:**
- Create: `scripts/README.md`, `scripts/requirements.txt`

**Interfaces:**
- Consumes: nothing.
- Produces: documentation only.

- [ ] **Step 1: Write `scripts/requirements.txt`**

`transformers` from git main (the `qwen3_5` architecture is not in a release at the time of writing), `torch>=2.6`, `peft>=0.13`, `datasets>=3`, `accelerate>=1.0`, `evaluate`, `rouge_score`, `pandas`, `numpy`, `pillow`, `torchvision`, `pytest`.

- [ ] **Step 2: Write `scripts/README.md`**

Cover: why `scripts/` exists and that it does not import `Pipeline/`; the `HF_TOKEN` export; the three commands plus `smoke_test.py`; the artifact map from the spec; the MPS notes (fp32, no bf16, no bitsandbytes, `PYTORCH_ENABLE_MPS_FALLBACK=1`, `sdpa`); the chat-template/thinking-off framing and that ROUGE numbers are **not** comparable to the README's Llama table with a one-line reason; the prompt/output boundary caveat.

- [ ] **Step 3: Verify the requirements resolve**

Run: `python -m pip install --dry-run -r scripts/requirements.txt` in a scratch venv; record any resolution conflict. Expected: resolves (transformers from git requires network).

---

### Task 9: Full verification and handover

**Files:** none created.
- [ ] **Step 1:** Run the whole suite: `cd scripts && python -m pytest tests -v` — record pass/skip counts.
- [ ] **Step 2:** `python -m py_compile scripts/*.py` — expect no output.
- [ ] **Step 3:** In a scratch venv with torch, run a tiny random-init `Qwen3_5ForConditionalGeneration` through `run_training` on 4 synthetic rows, using the real Qwen tokenizer, to prove `train.py` runs end to end (this is the substitute for the real 4B run, which needs the Mac).
- [ ] **Step 4:** Render the real chat template with the real tokenizer for one report; confirm it ends with ` thinking\n\n</think>\n\n`, that `eos_token_id == 248046`, `pad_token_id == 248044`, and that `tokenizer(text).input_ids` does not prepend a BOS.
- [ ] **Step 5:** Write the handover message: what changed, what was run with what output, what remains unverified (real 4B weights, MPS kernels, MPS Trainer path) and the exact command the user should run first on the Mac.

---

## Self-Review

- **Spec coverage:** chat-template framing → Task 2; parity tokenization/masking → Task 2; LoRA regex + params → Task 3; dtype/device policy → Tasks 1, 3; env-only token → Task 1; artifacts and dirs → Tasks 1, 4, 5, 6; smoke test → Task 7; README/requirements → Task 8; verification claims → Task 9. No gaps.
- **Step scan:** every test step carries its assertions; every code step carries signatures and the decisions the spec pins.
- **Type consistency:** `RunConfig` field names, `preprocess_input_data`/`preprocess_output_data` signatures, and the `{"input_ids", "labels"}` pickle shape are used identically in Tasks 2, 4, 5, 6.
- **Review Focus:** each of the five lines has a test in the owning task (Task 4 sizes/disjointness, Task 2 mask alignment, Task 2 empty relations, Task 2 truncation, Task 6 missing adapter dir).
- **Correction found during review:** `scripts/evaluate.py` would shadow the pip `evaluate` package when imported from `scripts/`, so the implementation lives in `evaluate_script.py` with `evaluate.py` as a thin entry point. Task 6 reflects this.
