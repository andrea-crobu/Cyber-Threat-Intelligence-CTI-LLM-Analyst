from types import SimpleNamespace

import pandas as pd
import pytest


THINK_OPEN = "<" + "think>"
THINK_CLOSE = "</" + "think>"


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
            {
                "messages": messages,
                "tokenize": tokenize,
                "add_generation_prompt": add_generation_prompt,
                **kwargs,
            }
        )
        text = (
            f"<|im_start|>{messages[0]['role']}\n{messages[0]['content']}"
            "<|im_end|>\n<|im_start|>assistant\n"
        )
        if kwargs.get("enable_thinking") is False:
            text += f"{THINK_OPEN}\n\n{THINK_CLOSE}\n\n"
        return text if not tokenize else [self._id(word) for word in text.split()]

    def __call__(self, text, truncation=False, max_length=None):
        self.encode_calls.append({"truncation": truncation, "max_length": max_length})
        ids = [self._id(word) for word in text.split()]
        if truncation and max_length is not None:
            ids = ids[:max_length]
        return SimpleNamespace(input_ids=ids)

    def decode(self, ids, skip_special_tokens=False):
        inverse = {value: key for key, value in self.vocab.items()}
        if skip_special_tokens:
            ids = [i for i in ids if int(i) not in (self.pad_token_id, self.eos_token_id)]
        return " ".join(inverse.get(int(i), "?") for i in ids)


@pytest.fixture
def stub_tokenizer():
    return StubTokenizer()


@pytest.fixture
def report_frame():
    return pd.DataFrame(
        [
            {
                "text": "NFlog and PoisonIvy were used by DragonOK.",
                "entities": str(
                    [
                        {"id": 0, "label": "malware", "start_offset": 0, "end_offset": 5},
                        {"id": 1, "label": "malware", "start_offset": 10, "end_offset": 19},
                        {"id": 2, "label": "threat-actor", "start_offset": 33, "end_offset": 41},
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
                "entities": str(
                    [{"id": 0, "label": "threat-actor", "start_offset": 0, "end_offset": 7}]
                ),
                "relations": "[]",
                "diagnosis": "A threat actor targets finance.",
            },
        ]
    )


class SavableStubTokenizer(StubTokenizer):
    """Stub that can also write itself to disk, like a real tokenizer."""

    def save_pretrained(self, directory):
        import pathlib

        pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
        (pathlib.Path(directory) / "tokenizer_config.json").write_text("{}")


@pytest.fixture
def tiny_adapter(tmp_path):
    """A tiny random Qwen3.5 checkpoint plus a LoRA adapter with non-zero weights."""
    torch = pytest.importorskip("torch")
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration

    config = Qwen3_5Config(
        text_config=dict(
            num_hidden_layers=1, layer_types=["full_attention"], hidden_size=16,
            num_attention_heads=1, num_key_value_heads=1, head_dim=8, intermediate_size=32,
            linear_num_key_heads=1, linear_num_value_heads=1, linear_key_head_dim=8,
            linear_value_head_dim=8, max_position_embeddings=128, mtp_num_hidden_layers=0,
            vocab_size=1024,
        ),
        vision_config=dict(
            depth=1, hidden_size=16, intermediate_size=16, num_heads=1,
            out_hidden_size=16, num_position_embeddings=4, patch_size=16,
        ),
    )
    torch.manual_seed(0)
    base = Qwen3_5ForConditionalGeneration(config)
    base_dir = tmp_path / "base"
    base.save_pretrained(base_dir)

    from config import DEFAULT_LORA_TARGET_MODULES

    lora = LoraConfig(
        r=4, lora_alpha=8, target_modules=DEFAULT_LORA_TARGET_MODULES,
        init_lora_weights=False, task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(base, lora)
    adapter_dir = tmp_path / "out" / "peft_model_CTI"
    peft_model.save_pretrained(adapter_dir)
    return {"base_dir": base_dir, "adapter_dir": adapter_dir, "out_dir": tmp_path / "out", "peft_model": peft_model}
