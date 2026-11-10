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
