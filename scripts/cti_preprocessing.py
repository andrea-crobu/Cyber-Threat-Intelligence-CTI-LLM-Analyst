"""Vendored CTI preprocessing: the project's prompt, framed as a Qwen chat turn.

The notebooks build a raw prompt string and mask it with -100 so the loss covers
only the answer. This module keeps that scheme and changes one thing: the prompt
is rendered through the tokenizer's chat template with thinking disabled, because
Qwen3.5 is an instruction-tuned model that expects <|im_start|>/<|im_end|> turns.
"""

from __future__ import annotations

import ast

import pandas as pd

PROMPT_INTRO = "You are a skilled AI Agent capable of doing CTI Analysis.\n\nGiven this threat report:\n"
PROMPT_OUTRO = (
    "\n\nYou will extract the main entities and their relations; "
    "finally, you will generate a diagnosis of the threat.\n"
)
REQUIRED_COLUMNS = ("text", "entities", "relations", "diagnosis")
IGNORE_INDEX = -100


class CTIPreprocessor:
    """Turns threat reports into token-id sequences and masked label sequences."""

    def __init__(self, tokenizer, max_length: int | None = None, enable_thinking: bool = False):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.enable_thinking = enable_thinking

    def build_prompt_text(self, report: str) -> str:
        return f"{PROMPT_INTRO}{report}{PROMPT_OUTRO}"

    def format_input_prompt(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        def render(row: pd.Series) -> pd.Series:
            messages = [{"role": "user", "content": self.build_prompt_text(row["text"])}]
            rendered = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
            return pd.Series({"prompt": rendered})

        return dataframe.apply(render, axis=1)

    def process_entries_and_relations(self, dataframe_row: pd.Series) -> pd.Series:
        # Work on a dict: pandas 3 string columns reject list values, so the raw
        # entity/relation lists must not be assigned back into the Series.
        values = dataframe_row.to_dict()
        entities = {}
        formatted_entities = []
        for entity in self._parse_literal(values["entities"], "entities"):
            name = values["text"][entity["start_offset"] : entity["end_offset"]]
            entities[entity["id"]] = {"name": name, "label": entity["label"]}
            formatted_entities.append(f"{name} ({entity['label']})")

        formatted_relations = []
        for relation in self._parse_literal(values["relations"], "relations"):
            source = entities.get(relation["from_id"])
            target = entities.get(relation["to_id"])
            if source is not None and target is not None:
                formatted_relations.append(
                    f"{source['name']} to {target['name']} ({relation['type']})"
                )

        return pd.Series(
            {
                **values,
                "entities": ", ".join(formatted_entities),
                "relations": ", ".join(formatted_relations),
            }
        )

    def format_output(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        def render(row: pd.Series) -> pd.Series:
            return pd.Series(
                {
                    "output": (
                        f"\nEntities: {row['entities']}"
                        f"\nRelations: {row['relations']}"
                        f"\nDiagnosis: {row['diagnosis']}"
                    )
                }
            )

        return dataframe.apply(render, axis=1)

    def tokenize(self, text: str) -> list[int]:
        encoded = self.tokenizer(
            text,
            truncation=self.max_length is not None,
            max_length=self.max_length,
        )
        return list(encoded.input_ids)

    def eos_ids(self) -> list[int]:
        return self.tokenize(self.tokenizer.eos_token)

    def preprocess_input_data(self, dataframe: pd.DataFrame, for_training: bool = True) -> list[list[int]]:
        self.validate_dataframe(dataframe)
        processed = self._process(dataframe)
        prompts = self.format_input_prompt(processed)["prompt"]

        if not for_training:
            return [self.tokenize(prompt) for prompt in prompts]

        outputs = self.format_output(processed)["output"]
        eos = self.eos_ids()
        return [
            self.tokenize(prompt) + self.tokenize(output) + eos
            for prompt, output in zip(prompts, outputs)
        ]

    def preprocess_output_data(self, dataframe: pd.DataFrame) -> list[list[int]]:
        self.validate_dataframe(dataframe)
        processed = self._process(dataframe)
        prompts = self.format_input_prompt(processed)["prompt"]
        outputs = self.format_output(processed)["output"]
        eos = self.eos_ids()

        return [
            [IGNORE_INDEX] * len(self.tokenize(prompt)) + self.tokenize(output) + eos
            for prompt, output in zip(prompts, outputs)
        ]

    def decode(self, ids: list[int], preserve_special_tokens: bool = False) -> str:
        return self.tokenizer.decode(
            [token for token in ids if token != IGNORE_INDEX],
            skip_special_tokens=not preserve_special_tokens,
        )

    def _process(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        return dataframe.apply(self.process_entries_and_relations, axis=1)

    @staticmethod
    def _parse_literal(value, column: str) -> list:
        if not isinstance(value, str):
            return value
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError) as error:
            raise ValueError(f"Column '{column}' is not a Python literal: {value!r}") from error
        if not isinstance(parsed, list):
            raise ValueError(f"Column '{column}' must hold a list, got {type(parsed).__name__}")
        return parsed

    @staticmethod
    def validate_dataframe(dataframe: pd.DataFrame) -> None:
        missing = [column for column in REQUIRED_COLUMNS if column not in dataframe.columns]
        if missing:
            raise ValueError(f"Dataframe missing required columns: {', '.join(missing)}")
