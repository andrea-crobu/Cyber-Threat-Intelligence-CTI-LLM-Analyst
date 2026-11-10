# CTI pipeline on Qwen3.5-4B — `scripts/` design

Date: 2026-11-10
Status: awaiting approval
Size: small change (six files, no product decision left open)

## Intent

Replicate the three-notebook CTI pipeline — dataset preparation → LoRA fine-tune →
ROUGE evaluation — as runnable scripts under `scripts/`, swapping
`meta-llama/Llama-3.2-3B-Instruct` for `Qwen/Qwen3.5-4B`, to run on an Apple
silicon Mac Studio M3 Ultra (MPS).

## Goals

- Same dataset, same 80/10/10 split, same prompt wording, same masked-label scheme.
- Prompt framed with Qwen's chat template, thinking disabled.
- LoRA fine-tune, then base-vs-fine-tuned ROUGE on the 48-row test split, same
  four metrics as the README table.
- One command per phase; every path a flag with a sane default; no Colab/Drive
  assumptions.
- Runs on MPS, also on CUDA and CPU.

## Non-goals

- Not changing the data, the split ratios, or the prompt wording.
- No quantization, no vLLM/SGLang serving, no chat UI.
- Not touching the notebooks or the `Pipeline` submodule.

## Anti-goals

- A run whose numbers silently look comparable to the README table but are not
  (different framing or masking).
- Failing on the Mac because of bf16, or filling memory with a full fine-tune.
- Claiming success without a smoke test on the actual hardware.

## Constraints

- `Qwen/Qwen3.5-4B` is `Qwen3_5ForConditionalGeneration` — a vision-language
  model. Needs `transformers` from `main` (4.57+), plus torchvision/pillow.
  `AutoModelForCausalLM` cannot load it.
- Tokenizer: `eos = <|im_end|>` (248046), `pad = <|endoftext|>` (248044), distinct;
  `add_bos_token: false`; `model_max_length: 262144`.
- MPS: no bfloat16, no bitsandbytes; flash-attention unavailable, `sdpa` instead;
  the gated-delta-rule fallback calls `torch.linalg.solve_triangular`, which needs
  `PYTORCH_ENABLE_MPS_FALLBACK=1`.
- Dataset: 476 rows → 380 train / 48 validation / 48 test.
- This development machine has no GPU and no torch, so training itself cannot be
  executed here.

## Approaches considered

**A. Thin scripts over the `Pipeline` submodule** — subclass `CTI_Preprocessor`,
override the prompt framing. Smallest diff, one source of truth for preprocessing.
Rejected: you chose a self-contained folder.

**B. Self-contained `scripts/`** — vendor the preprocessing logic. Runs on a
machine that never cloned `Pipeline`; costs a second copy that can drift from the
notebooks. **Chosen.**

**C. Single CLI with subcommands** — one entry point, all flags centralised.
Rejected: less faithful to "the notebooks as scripts".

## Decisions

1. **Prompt framing.** Raw project prompt wrapped via
   `apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
   add_generation_prompt=True, enable_thinking=False)`, so the assistant turn
   opens with `<|im_start|>assistant\n thinking\n\n</think>\n\n`.
2. **Tokenization parity with the notebooks.** Prompt ids and output ids are
   tokenized separately and concatenated; labels are
   `[-100] * len(prompt_ids) + output_ids + eos_ids`. Known caveat: the
   prompt/output boundary is not a single decode of the whole string, so the first
   output token can merge with the prompt's trailing newline. Kept for faithfulness;
   documented in the README rather than fixed.
3. **Labels carry `<|im_end|>` as EOS**, matching the notebook's appended eos.
4. **LoRA target.** Text-decoder linears only, by regex:
   `.*language_model\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj|in_proj_qkv|in_proj_z|in_proj_b|in_proj_a|out_proj)$`.
   `r=32`, `alpha=32`, `dropout=0.05`, `task_type=CAUSAL_LM`. Vision tower excluded:
   this task is text-only, and the notebook's `["q_proj","v_proj"]` would miss the
   24 recurrent layers entirely.
5. **dtype policy.** fp32 for training on MPS/CPU, bf16 only where supported,
   fp32 for generation. `attn_implementation="sdpa"`.
6. **Training defaults** (notebook parity): batch 1, 1 epoch, `AdamW(lr=1e-3)` on
   LoRA params only, `eval_steps=100`, `logging_steps=100`, seed 42,
   `DataCollatorForSeq2Seq(label_pad_token_id=-100)`.
7. **Secrets.** HF token read from `HF_TOKEN` env var only; no placeholder strings
   written into files.
8. **Artifacts.** `data/dataset_CTI_qwen3.5-4B.pkl`, `outputs/peft_model_CTI/`,
   `outputs/model_predictions.csv`, `outputs/rouge_scores.json`. All under
   `--data-dir` / `--output-dir`.
9. **Smoke test first.** `smoke_test.py` loads the model, runs one forward+backward
   step on two samples and reports step time and memory, so the full run is never
   the first thing tried on new hardware.

## Files

```
scripts/
├── README.md              # run order, MPS notes, artifact map
├── requirements.txt       # transformers @ main, torch, peft, datasets, evaluate…
├── config.py              # RunConfig dataclass + argparse binding
├── cti_preprocessing.py   # vendored preprocessing with chat-template framing
├── model_utils.py         # device/dtype policy, model+tokenizer loaders, LoRA
├── prepare_dataset.py     # phase 1
├── train.py               # phase 2
├── evaluate.py            # phase 3
├── smoke_test.py          # one-step forward/backward on new hardware
└── tests/
    ├── conftest.py        # stub tokenizer, no torch required
    └── test_preprocessing.py
```

## Left to the builder

Flag names, test assertions, README wording, and the exact shape of the ROUGE
report. Nothing here changes what gets built.

## Verification

- `pytest scripts/tests` — preprocessing, masking, eos and template assertions
  against a stub tokenizer. Runs here, no torch needed.
- `python -m py_compile scripts/*.py`.
- `transformers` import + real tokenizer rendering of the chat template in a
  throwaway venv here, to confirm the template output and special-token ids.
- Training, generation and ROUGE: **unverified until you run `smoke_test.py` on the
  Mac.** I will say so explicitly and hand over the exact command.
