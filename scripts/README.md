# CTI pipeline on Qwen3.5-4B

Runnable version of the three notebooks at the repository root, with
`Qwen/Qwen3.5-4B` in place of `meta-llama/Llama-3.2-3B-Instruct`.

The folder is **self-contained**: it does not import `Pipeline/` or any notebook.
The preprocessing the notebooks take from `Pipeline/preprocessing/CTI_Preprocessor.py`
is vendored in `cti_preprocessing.py` with one change — the prompt is rendered
through Qwen's chat template.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r scripts/requirements.txt
export HF_TOKEN=hf_...            # only if the checkpoint needs a token; never hard-coded
```

`Qwen/Qwen3.5-4B` is public, so `HF_TOKEN` is optional for it. Any gated model
requires the token; it is read from the env var named by `--hf-token-env`
(default `HF_TOKEN`) and never written to a file.

Then, from the repository root:

```bash
python scripts/prepare_dataset.py            # → data/dataset_CTI_qwen3.5-4B.pkl
python scripts/smoke_test.py                 # one forward+backward, on real hardware
python scripts/train.py                      # → outputs/peft_model_CTI/
python scripts/run_evaluation.py             # → outputs/model_predictions.csv, outputs/rouge_scores.json
python scripts/merge_adapter.py              # → outputs/merged_model_CTI/ (standalone model, no peft needed)
```

Run `smoke_test.py` first on a new machine: it loads the checkpoint, runs one
training step on two real examples, and prints the step time so the full run can
be projected before you commit to it.

Every flag has a default; `--help` on any script lists them all. The common ones:

| flag | default | notes |
|------|---------|-------|
| `--model-id` | `Qwen/Qwen3.5-4B` | any causal or image-text checkpoint |
| `--data-dir` / `--output-dir` | `data` / `outputs` | relative to where you run the script |
| `--device` | `auto` | `auto`, `mps`, `cuda`, `cpu` |
| `--dtype` | `auto` | `float32` on mps/cpu, `bfloat16` on cuda |
| `--max-length` | unset | caps each side of the sequence; unset = no truncation |
| `--lora-r` / `--lora-alpha` | 32 / 32 | |
| `--num-train-epochs` | 1.0 | 476 rows → 380 steps per epoch at batch 1 |

## Artifacts

| path | written by |
|------|-----------|
| `data/dataset_CTI_qwen3.5-4B.pkl` | `prepare_dataset.py` |
| `outputs/training/` | `train.py` (Trainer output dir) |
| `outputs/peft_model_CTI/` | `train.py` (adapter + tokenizer) |
| `outputs/merged_model_CTI/` | `merge_adapter.py` (weights + tokenizer) |
| `outputs/model_predictions.csv` | `run_evaluation.py` |
| `outputs/rouge_scores.json` | `run_evaluation.py` |

## Three things that behave differently from the notebooks

1. **Prompt framing.** The project's prompt text is unchanged, but it is wrapped
   with `apply_chat_template(..., add_generation_prompt=True, enable_thinking=False)`,
   so the model sees a real user turn and an assistant turn that opens with
   an open-and-immediately-closed thinking block (empty, then a blank line). Qwen3.5
   thinks by default and the chat template has no `/think` switch; handing the model
   an already-closed thinking block is how non-thinking mode is requested.

2. **The numbers are not comparable to the README table.** That table came from
   Llama-3.2-3B-Instruct trained on raw, unframed prompts. Different base model,
   different framing, different tokenizer, different LoRA targets: the ROUGE
   scores here are a fresh baseline for Qwen3.5-4B, not a re-measurement of it.

3. **LoRA covers more of the model.** The notebook adapts `q_proj`/`v_proj`, which
   on Qwen3.5 would reach only the 8 full-attention layers. The default regex also
   adapts the 24 gated-delta-net layers (`in_proj_qkv`, `in_proj_z`, `in_proj_b`,
   `in_proj_a`, `out_proj`) and every MLP, and deliberately skips the vision tower.

## Known caveat: the prompt/output seam

Prompt ids and output ids are tokenized separately and concatenated, exactly as
the notebooks do. The first output token can therefore merge with the prompt's
trailing newline when a tokenizer sees them together. It is kept for parity with
the notebooks rather than fixed silently; tightening it would change what the
loss sees.

## Merging the adapter

`merge_adapter.py` loads the base model, applies `outputs/peft_model_CTI/`, folds the
LoRA deltas into the weights and saves a plain checkpoint plus the tokenizer. Load it
with `AutoModelForImageTextToText.from_pretrained("outputs/merged_model_CTI")`.

* It refuses to overwrite an existing merged folder; pass `--overwrite` or `--merged-dirname`.
* The merge runs in the dtype you pass (`fp32` by default on the Mac, ~16 GB on disk). Add
  `--dtype fp16` for a half-size file, accepting a small rounding change versus the fp32 merge.
* Merging bakes the adapter in permanently: keep `outputs/peft_model_CTI/` if you want to
  retrain or compare against the base model.

## Apple silicon notes

* `bfloat16` is not supported on MPS. The default is fp32 for training and
  generation; `--dtype fp16` is accepted and faster but less stable for training.
* No bitsandbytes on MPS, so no QLoRA: the base weights stay in full precision.
  4B in fp32 is ~16 GB, plus activations. Budget ~32 GB of unified memory.
* `PYTORCH_ENABLE_MPS_FALLBACK=1` is set by `model_utils` before torch is
  imported. The gated-delta-rule fallback needs `torch.linalg.solve_triangular`,
  which has no MPS kernel and otherwise fails on the first forward pass.
* `attn_implementation="sdpa"`, not flash-attention.
* Training is slower than on CUDA: the recurrent layers run chunk by chunk on CPU.
  One epoch of 380 steps at batch 1 is tens of minutes, not minutes.

## Tests

```bash
cd scripts && python -m pytest -v
```

The suite runs without a GPU. Tests that need torch skip themselves when it is
absent (`pytest.importorskip`). Nothing in the suite downloads the 4B checkpoint.

## Layout

```
config.py            RunConfig dataclass, CLI flags, device/dtype policy
cti_preprocessing.py prompt building, chat template, entity/relation formatting, masking
model_utils.py       device/dtype resolution, tokenizer/model loaders, LoRA
prepare_dataset.py   phase 1
train.py             phase 2
merge_adapter.py     fold the adapter into the base weights
evaluate_script.py   phase 3 logic; run_evaluation.py is the entry point, named so it
                     cannot shadow the pip `evaluate` package
smoke_test.py        one forward+backward on new hardware
tests/               pytest suite with a word-level stub tokenizer
```
